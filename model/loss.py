import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace
from utils.util import gather_tensors
from configs.cfgparser import Config


# --- Cross-Entropy (pixel-wise) Losses ---
class SegCE(nn.Module):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates Cross-Entropy loss on the segmentation logits.

        Parameters
        ----------
        inputs : SimpleNamespace
            An object containing the following attributes:
            - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model
            - `seg_gt` | torch.Tensor (B, C, H, W) | 1-hot encoded ground truth segmentation mask.

        Returns
        -------
        loss : torch.Tensor
            The computed Segmentation Cross-Entropy loss.

        Raises
        ------
        ValueError
            If `inputs.seg_logits` or `inputs.seg_gt` is None.
        """

        if inputs.seg_logits is None or inputs.seg_gt is None:
            raise ValueError(
                "Inputs must contain 'seg_logits' and 'seg_gt' attributes."
            )

        # (B, C, H, W) → (B, H, W)
        target = inputs.seg_gt.argmax(dim=1)

        # compute CE loss → scalar
        loss = F.cross_entropy(inputs.seg_logits, target)
        return loss


class WeightedSegCE(nn.Module):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates Weighted Cross-Entropy loss on the segmentation logits.
        The weights are derived from the ground truth Signed Distance Map (`sdm_gt`).

        Parameters
        ----------
        inputs : SimpleNamespace
            An object containing the following attributes:
            - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
            - `seg_gt` | torch.Tensor (B, C, H, W) | The ground truth segmentation mask.
            - `sdm_gt` | torch.Tensor (B, 1, H, W) | The ground truth Signed Distance Map.

        Returns
        -------
        loss : torch.Tensor
            The computed weighted binary cross-entropy loss.

        Raises
        ------
        ValueError
            If `inputs.seg_logits`, `inputs.seg_gt`, or `inputs.sdm_gt` is None.
        """

        if inputs.seg_logits is None or inputs.seg_gt is None or inputs.sdm_gt is None:
            raise ValueError(
                "Inputs must contain 'seg_logits', 'seg_gt', and 'sdm_gt' attributes."
            )

        # (B, C, H, W) → (B, H, W)
        target = inputs.seg_gt.argmax(dim=1)

        # build per-pixel weight map (B, H, W)
        weight_map = ((2 - inputs.sdm_gt.abs()) ** 2).squeeze(1)

        # compute raw per-pixel CE loss → (B, H, W)
        per_pixel_loss = F.cross_entropy(inputs.seg_logits, target, reduction="none")

        # apply per-pixel weights; average over batch & spatial dims → scalar
        loss = torch.mean(per_pixel_loss * weight_map)
        return loss


class ClsCE(nn.Module):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates cross-entropy loss on the classification logits.

        Parameters
        ----------
        inputs : SimpleNamespace
            An object containing the following attributes:
            - `cls_logits` | torch.Tensor (B, C) | The classification logits output by the model.
            - `cls_gt` | torch.Tensor (B, ) | The ground truth class labels.

        Returns
        -------
        loss : torch.Tensor
            The computed cross-entropy loss.

        Raises
        ------
        ValueError
            If `inputs.cls_logits` or `inputs.cls_gt` is None.
        """

        if inputs.cls_logits is None or inputs.cls_gt is None:
            raise ValueError(
                "Inputs must contain 'cls_logits' and 'cls_gt' attributes."
            )

        loss = F.cross_entropy(inputs.cls_logits, inputs.cls_gt)
        return loss


# --- MAE (sdm-based) Losses ---
class BaseSDMLoss(nn.Module):
    """
    Base class for SDM-based losses. Handles extracting the ground truth,
    filtering the batch by the target class (cls_gt == 2), and applying the
    tanh activation to the logits.
    """
    def __init__(self, boundary_id: int):
        """Initializes the BaseSDMLoss class.
        
        Parameters
        ----------
        boundary_id : int
            The class ID corresponding to the boundary class in `cls_gt`. Only samples
            with `cls_gt` equal to this ID will contribute to the loss computation.
        """
        
        super().__init__()
        self.boundary_id = boundary_id

    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """Calculates the loss based on the filtered signed distance map.
        
        Parameters
        ----------
        inputs : SimpleNamespace
            An object containing the following attributes:
            - `sdm_logits` | torch.Tensor (B, 1, H, W) | The segmentation logits output by the model.
            - `sdm_gt` | torch.Tensor (B, 1, H, W) | The ground truth signed distance map.
            - `cls_gt` | torch.Tensor (B, ) | The ground truth class labels.
            - `seg_gt` | torch.Tensor (B, C, H, W) | The ground truth segmentation mask.
            
        Returns
        -------
        loss : torch.Tensor
            The computed loss based on the filtered signed distance map.
                
        Raises
        ------
        ValueError
            If any of the required attributes are missing from `inputs`.
        """
        self._validate_inputs(inputs)
        
        idx = (inputs.cls_gt == self.boundary_id)
        if not idx.any():
            return inputs.sdm_logits.sum() * 0.0

        # logits (B, 1, H, W) → SDM (B, 1, H, W), filtered by idx
        sdm_pd = torch.tanh(inputs.sdm_logits)[idx]
        sdm_gt = inputs.sdm_gt[idx]
        seg_gt = inputs.seg_gt[idx]

        return self._compute_loss(sdm_pd, sdm_gt, seg_gt)
    
    def _validate_inputs(self, inputs: SimpleNamespace):
        if getattr(inputs, 'sdm_logits', None) is None or \
           getattr(inputs, 'sdm_gt', None) is None or \
           getattr(inputs, 'cls_gt', None) is None or \
           getattr(inputs, 'seg_gt', None) is None:
            raise ValueError(
                "Inputs must contain 'sdm_logits', 'sdm_gt', 'cls_gt', 'seg_gt' attributes."
            )

    def _compute_loss(self, 
                      sdm_pd: torch.Tensor, 
                      sdm_gt: torch.Tensor, 
                      seg_gt: torch.Tensor
                      ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement _compute_loss.")
  
    
class MAE(BaseSDMLoss):
    def __init__(self, boundary_id: int):
        super().__init__(boundary_id=boundary_id)
        
    def _compute_loss(self, 
                      sdm_pd: torch.Tensor, 
                      sdm_gt: torch.Tensor, 
                      seg_gt: torch.Tensor) -> torch.Tensor:
        """Calculates mean absolute error loss on the filtered segmentation logits."""
        return torch.mean(torch.abs(sdm_pd - sdm_gt))
    
    
class ClampedMAE(BaseSDMLoss):
    def __init__(self, boundary_id: int, delta: float):
        super().__init__(boundary_id=boundary_id)
        self.delta = delta

    def _compute_loss(self, 
                      sdm_pd: torch.Tensor, 
                      sdm_gt: torch.Tensor, 
                      seg_gt: torch.Tensor
                      ) -> torch.Tensor:
        """Calculates clamped mean absolute error loss on the filtered segmentation logits."""
        # clamp the pd/gt SDMs to [-δ; δ] range
        loss = torch.abs(
            torch.clamp(sdm_pd, -self.delta, self.delta)
            - torch.clamp(sdm_gt, -self.delta, self.delta)
        )

        # average over batch & spatial dims → scalar
        return torch.mean(loss / (2 * self.delta))
    
    
class SignMAE(BaseSDMLoss):
    def __init__(self, boundary_id: int, k: int = 1000, q: int = 4):
        super().__init__(boundary_id=boundary_id)
        self.k: int = k
        self.q: int = q

    def _compute_loss(self, 
                      sdm_pd: torch.Tensor, 
                      sdm_gt: torch.Tensor, 
                      seg_gt: torch.Tensor
                      ) -> torch.Tensor:
        """Calculates a custom Sign loss based on the filtered signed distance map."""
        # calculate the Sign loss component → scalar
        sign = torch.sigmoid(sdm_pd * self.k * (1 - 2 * seg_gt))
        sign_loss = torch.mean(self.q * sign * (sdm_pd**2))

        # calculate the MAE loss component → scalar
        mae_loss = torch.mean(torch.abs(sdm_pd - sdm_gt))

        # combine the two components → scalar
        return (sign_loss + mae_loss) / 2


# --- DICE/IoU (area) Losses ---
class BaseDiceLoss(nn.Module):
    def __init__(self, k: int = 1000, eps: float = 1e-6):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, inputs: SimpleNamespace):
        raise NotImplementedError

    def compute_loss(self, gt: torch.Tensor, pd: torch.Tensor) -> torch.Tensor:
        """
        Computes the DICE loss between the ground truth and predicted masks.

        Parameters
        ----------
        gt : torch.Tensor (B, C, H, W).
            1-hot encoded ground truth mask.

        pd : torch.Tensor (B, C, H, W).
            Probabilistic (softmax) predicted mask.

        Returns
        -------
        loss: torch.Tensor
            The computed DICE loss.
        """

        # sum over batch and spatial dimensions
        # pd/gt (B, C, H, W) → per_class_dice (C,)
        dims = (0, 2, 3)
        numer = torch.sum(gt * pd, dims)
        denom = torch.sum(gt, dims) + torch.sum(pd, dims)
        per_class_dice = (2 * numer + self.eps) / (denom + self.eps)

        # avoid DICE computation on empty classes
        # loss (C, ) → scalar
        valid = torch.sum(gt, dims) > 0
        if valid.any():
            loss = 1.0 - per_class_dice[valid].mean()
        else:
            loss = torch.tensor(0.0, device=gt.device)
        return loss


class SoftDice(BaseDiceLoss):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates Soft (probabilistic) DICE loss on the segmentation logits.
        Every voxel contributes to the overlap score, so gradients
        remain informative even when the network is undecided.

        Parameters
        ----------
        inputs : SimpleNamespace
            An object containing the following attributes:
            - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
            - `seg_gt` | torch.Tensor (B, C, H, W) | The ground truth segmentation mask.

        Returns
        -------
        loss : torch.Tensor
            The computed soft DICE loss.

        Raises
        ------
        ValueError
            If `inputs.seg_logits` or `inputs.seg_gt` is None.
        """

        if inputs.seg_logits is None or inputs.seg_gt is None:
            raise ValueError(
                "Inputs must contain 'seg_logits' and 'seg_gt' attributes."
            )

        # logits (B, C, H, W) → probabilities (B, C, H, W)
        mask_pd = F.softmax(inputs.seg_logits, dim=1)

        # compute DICE loss; pd/gt (B, C, H, W) → scalar
        return self.compute_loss(inputs.seg_gt, mask_pd)


class HardDice(BaseDiceLoss):
    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """Dice with temperature-scaled softmax to sharpen predictions.

        Calculates Hard (discrete) DICE loss on the segmentation logits.
        Applies a steep sigmoid, so that the activation curve approaches a step function.
        The narrow transition zone encourages the network to emit probabilities close to 0 or 1,
        effectively training it to behave as if hard thresholding had already been applied.
        This sharpens object contours but can make optimisation less forgiving.

        Parameters
        ----------
        inputs : SimpleNamespace
            An object containing the following attributes:
            - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
            - `seg_gt` | torch.Tensor  (B, C, H, W) | The ground truth segmentation mask.

        Returns
        -------
        loss : torch.Tensor
            The computed hard DICE loss.

        Raises
        ------
        ValueError
            If `inputs.seg_logits` or `inputs.seg_gt` is None.
        """

        if inputs.seg_logits is None or inputs.seg_gt is None:
            raise ValueError(
                "Inputs must contain 'seg_logits' and 'seg_gt' attributes."
            )

        # logits (B, C, H, W) → probabilities (B, C, H, W)
        mask_pd = F.softmax(self.k * inputs.seg_logits, dim=1)

        # compute DICE loss; pd/gt (B, C, H, W) → scalar
        return self.compute_loss(inputs.seg_gt, mask_pd)


class IoU(nn.Module):
    def __init__(self, k: int = 1000, eps: float = 1e-6):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Calculates Intersection over Union (IoU) loss on the segmentation logits.

        Parameters
        ----------
        inputs : SimpleNamespace
            An object containing the following attributes:
            - `seg_logits` | torch.Tensor (B, C, H, W) | The segmentation logits output by the model.
            - `seg_gt` | torch.Tensor (B, C, H, W) | The ground truth segmentation mask.

        Returns
        -------
        loss : torch.Tensor
            The computed IoU loss.

        Raises
        ------
        ValueError
            If `inputs.seg_logits` or `inputs.seg_gt` is None.
        """

        if inputs.seg_logits is None or inputs.seg_gt is None:
            raise ValueError(
                "Inputs must contain 'seg_logits' and 'seg_gt' attributes."
            )

        # logits (B, C, H, W) → probabilities (B, C, H, W)
        mask_pd = F.softmax(self.k * inputs.seg_logits, dim=1)

        # sum over batch and spatial dimensions
        # pd/gt (B, C, H, W) → per_class_iou (C,)
        dims = (0, 2, 3)
        inter = torch.sum(inputs.seg_gt * mask_pd, dims)
        union = torch.sum(inputs.seg_gt, dims) + torch.sum(mask_pd, dims) - inter
        per_class_iou = (inter + self.eps) / (union + self.eps)

        # avoid IoU computation on empty classes
        # loss (C,) → scalar
        valid = torch.sum(inputs.seg_gt, dims) > 0
        if valid.any():
            loss = 1.0 - per_class_iou[valid].mean()
        else:
            loss = torch.tensor(0.0, device=inputs.seg_gt.device)
        return loss


# --- Combined Loss ---
class CombinedLoss(nn.Module):
    def __init__(
        self,
        losses: list[nn.Module],
        device: str,
        static_weights: list[int] | None):
        """
        Combines multiple loss functions into a single loss function.
        The combined loss is computed as a weighted sum of the individual losses.
        The weights can be either fixed or adaptive, depending on the `cfg.ADAPTIVE_WEIGHTS`.

        Parameters
        ----------
        losses : list[nn.Module]
            A list of loss functions to be combined.
            
        device : str
            The device on which the loss functions will be computed (e.g. `cuda:0`).
            
        static_weights : list[int] | None
            A list of static weights for each loss function. 
            If None, the weights will be learnable parameters.
        """

        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.device: str = device
        self.static_weights: list[int] | None = static_weights

        if self.static_weights:
            self.weights = torch.tensor(
                self.static_weights, dtype=torch.float32, device=self.device
            )
        else:
            self.weights = nn.Parameter(
                torch.zeros(len(self.losses), dtype=torch.float32, device=self.device)
            )

    def forward(self, inputs: SimpleNamespace) -> torch.Tensor:
        """
        Computes the combined loss based on the individual losses and their weights.

        Parameters
        ----------
        inputs : SimpleNamespace
            An inputs object containing the necessary attributes for the loss functions.

        Returns
        -------
        total_loss : torch.Tensor
            The computed combined loss.
        """

        # Collect individual loss values (N,) where N = len(self.losses)
        loss_values = torch.stack([loss(inputs) for loss in self.losses], dim=0)

        # Combine losses with weights; losses (N,) * weights (N,) → scalar
        if self.static_weights:
            total_loss = (loss_values * self.weights).sum() / self.weights.sum()
        else:            
            sp = F.softplus(self.weights) + 1.0
            total_loss = ((loss_values / (2 * sp**2)) + (torch.log(sp**2) / 2)).sum()
            total_loss = total_loss / loss_values.numel() 
        return total_loss


class Loss:
    def __init__(self, cfg: Config):
        """
        Initializes the loss function based on the `cfg.LOSS` loss;
        Can be a single loss or a combination of multiple losses.

        Segmentation losses:
        - `SoftDICE`: Soft DICE loss
        - `HardDICE`: Hard DICE loss
        - `IoU`: Intersection over Union loss
        - `SegCE`: Segmentation Cross-Entropy loss
        - `wSegCE`: Weighted Segmentation Cross-Entropy loss

        Boundary/SDM losses:
        - `MAE`: Mean Absolute Error loss
        - `cMAE`: Clamped Mean Absolute Error loss
        - `sMAE`: Signed Mean Absolute Error loss

        Classification losses:
        - `ClsCE`: Classification Cross-Entropy loss

        Parameters
        ----------
        cfg : Config
            Configuration object that contains the following attributes:
            - `.LOSS` (list[str]): List of loss function names to be used.
            - `.STATIC_WEIGHTS` (list[int] | None): List of static weights for balancing multiple losses.
            - `.CLAMP_DELTA` (float): The delta value for the Clamped MAE loss
            - `.SEG_CLASSES` (int): Corresponds to the class ID of the boundary class in `cls_gt`.
            - `.DEVICE` (str): The device on which the loss function will be computed (e.g. `cuda:0`).
            - `.WORLD_SIZE` (int): The number of GPUs used for training.
        """

        lnames: list[str] = cfg.LOSS
        static_weights: list[int] | None = cfg.STATIC_WEIGHTS
        delta: float = cfg.CLAMP_DELTA
        boundary_id = cfg.SEG_CLASSES
        
        self.device: str = cfg.DEVICE
        self.worldsize: int = cfg.WORLD_SIZE
        self.totalLoss: torch.Tensor = torch.tensor(
            0, dtype=torch.float32, device=self.device
        )
        loss_map = {
            "SoftDICE": SoftDice(),
            "HardDICE": HardDice(),
            "IoU": IoU(),
            "SegCE": SegCE(),
            "wSegCE": WeightedSegCE(),
            "ClsCE": ClsCE(),
            "MAE": MAE(boundary_id),
            "cMAE": ClampedMAE(boundary_id, delta),
            "sMAE": SignMAE(boundary_id),
        }

        if len(lnames) > 1:
            self.lfunc = CombinedLoss(
                [loss_map[lname] for lname in lnames],
                device=self.device,
                static_weights=static_weights,
            )
        else:
            self.lfunc = loss_map[lnames[0]]

    def update(self, logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        """
        Updates the loss function with the current batch of logits and ground truth values.

        Parameters
        ----------
        logits : dict[str, torch.Tensor]
            A dictionary containing the model's output logits.

        batch : dict[str, torch.Tensor]
            A dictionary containing the ground truth values.
        """

        inputs = SimpleNamespace(
            seg_logits=logits.get("seg"),
            sdm_logits=logits.get("sdm"),
            cls_logits=logits.get("cls"),
            seg_gt=batch.get("mask"),
            sdm_gt=batch.get("sdm"),
            cls_gt=batch.get("cls"),
        )

        self.loss = self.lfunc(inputs)
        self.totalLoss = self.totalLoss + self.loss.detach()

    def compute_avg(self, length: int) -> dict[str, float]:
        """
        Computes the average loss over the entire dataset.

        Parameters
        ----------
        length : int
            The number of batches in the dataset.

        Returns
        -------
        avgLoss : dict[str, float]
            A dictionary containing the average loss values.
        """

        avgLoss = {"loss": self.totalLoss / length}

        if self.worldsize > 1:
            avgLoss = gather_tensors(avgLoss, self.worldsize)

        return {k: round(v.item(), 4) for k, v in avgLoss.items()}
    
    def parameters(self):
        """
        Exposes the learnable parameters of the underlying loss function 
        (e.g., adaptive weights) to the optimizer.
        """

        if isinstance(self.lfunc, nn.Module):
            return self.lfunc.parameters()
        
        return iter([])

    def backprop(self):
        self.loss.backward()

    def reset(self):
        self.totalLoss: torch.Tensor = torch.tensor(
            0, dtype=torch.float32, device=self.device
        )