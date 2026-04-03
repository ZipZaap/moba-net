import torch
from utils.sdf import SDF
from utils.util import gather_tensors, logits_to_lbl, logits_to_msk
from configs.cfgparser import Config


class SegmentationMetrics:
    """
    Class for computing segmentation metrics.
    """

    def __init__(self, cfg: Config):
        """
        Initializes the SegmentationMetrics class with the configuration object.

        Parameters
        ----------
        cfg : Config
            Configuration object containing the following attributes:
            - `.INPUT_SIZE` (tuple[int, int]): Input image size (height, width).
            - `.SDM_KERNEL_SIZE` (int): Kernel size for SDM computation.
            - `.SDM_DISTANCE` (str): Distance type for SDM computation.
            - `.SDM_NORMALIZATION` (str): Normalization method for SDM computation.
            - `.CLAMP_DELTA` (float): Delta value for clamping the SDM during boundary metric computation.
            - `.CMA_COEFFICIENTS` (dict[str, int]): Coefficients for the Combined Mean Accuracy (CMA) calculation.
        """

        self.imsize: tuple[int, int] = cfg.INPUT_SIZE
        self.K: int = cfg.SDM_KERNEL_SIZE
        self.dist: str = cfg.SDM_DISTANCE
        self.norm: str = cfg.SDM_NORMALIZATION
        self.delta: float = cfg.CLAMP_DELTA
        self.coef: dict[str, int] = cfg.CMA_COEFFICIENTS
        self.eps: float = 1e-6

    def dice(self, pd_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice Similarity Coefficient (DSC) between predicted and ground truth masks.

        Parameters
        ----------
        pd_mask : torch.Tensor (B, C, H, W)
            1-hot encoded predicted mask.

        gt_mask : torch.Tensor (B, C, H, W)
            1-hot encoded ground truth mask.

        Returns
        -------
        dice : torch.Tensor
            Dice Similarity Coefficient.
        """

        # sum over batch and spatial dimensions
        # pd/gt (B, C, H, W) → per_class_dice (C,)
        dims = (0, 2, 3)
        numer = torch.sum(gt_mask * pd_mask, dims)
        denom = torch.sum(gt_mask, dims) + torch.sum(pd_mask, dims)
        per_class_dice = (2 * numer + self.eps) / (denom + self.eps)

        # filter out classes with no ground truth
        # per_class_dice (C,) → scalar
        valid = torch.sum(gt_mask, dims) > 0
        return per_class_dice[valid].mean()

    def iou(self, pd_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the Intersection over Union (IoU) between predicted and ground truth masks.

        Parameters
        ----------
        pd_mask : torch.Tensor (B, C, H, W)
            1-hot encoded predicted mask.

        gt_mask : torch.Tensor (B, C, H, W)
            1-hot encoded ground truth mask.

        Returns
        -------
        iou : torch.Tensor
            Intersection over Union.
        """

        # sum over batch and spatial dimensions
        # pd/gt (B, C, H, W) → per_class_iou (C,)
        dims = (0, 2, 3)
        inter = torch.sum(gt_mask * pd_mask, dims)
        union = torch.sum(gt_mask, dims) + torch.sum(pd_mask, dims) - inter
        per_class_iou = (inter + self.eps) / (union + self.eps)

        # avoid IoU computation on empty classes
        # per_class_iou (C,) → scalar
        valid = torch.sum(gt_mask, (0, 2, 3)) > 0
        return per_class_iou[valid].mean()

    def boundary(
        self, 
        gt_mask: torch.Tensor, 
        gt_sdm: torch.Tensor,
        pd_mask: torch.Tensor, 
        pd_sdm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Computes the boundary (SDM-based) metrics between predicted and ground truth masks.

        Parameters
        ----------
        pd_mask : torch.Tensor (B, C, H, W)
            1-hot encoded predicted mask.

        gt_mask : torch.Tensor (B, C, H, W)
            1-hot encoded ground truth mask.

        gt_sdm : torch.Tensor (B, 1, H, W)
            Ground truth Signed Distance Map.
            
        pd_sdm : torch.Tensor (B, 1, H, W) | None
            Predicted Signed Distance Map. If `None`, the SDM 
            will be computed from the predicted mask.

        Returns
        -------
        asd, hd95 : tuple[torch.Tensor, ...]
            A tuple containing the following distance metrics:
            - Average Symmetric Distance (ASD)
            - Hausdorff Distance (HD95)
        """

        if pd_sdm is None:
            # mask (B, C, H, W) → sdm (B, 1, H, W)
            pd_sdm = torch.abs(SDF.sdf(pd_mask, self.K, self.dist, self.norm))
        else:
            # alternatively, directly pass the predicted SDM from the model.
            pd_sdm = torch.abs(pd_sdm)
        
        gt_sdm = torch.abs(gt_sdm)

        # mask (B, C, H, W) → edges (B, 1, H, W)
        gt_edges = SDF.compute_sobel_edges(gt_mask).bool()
        pd_edges = SDF.compute_sobel_edges(pd_mask).bool()

        # gate the evaluation to a corridor around the GT boundary
        corridor = gt_sdm < self.delta
        gt_edges &= corridor
        pd_edges &= corridor

        # if edges are empty, default to max penalty
        fallback = torch.tensor(1.0, dtype=torch.float32, device=gt_sdm.device)

        # compute distances
        asd, hd95 = [], []
        for gE, pE, gS, pS in zip(gt_edges, pd_edges, gt_sdm, pd_sdm):
            d1, d2 = pS[gE], gS[pE]
            d = torch.cat((d1, d2))

            asd.append(fallback if torch.isnan(d).any() else d.median())
            hd95.append(fallback if torch.isnan(d).any() else torch.quantile(d, 0.95))

        # asd/hd95 (B,) → scalar
        asd = torch.stack(asd).mean()
        hd95 = torch.stack(hd95).mean()
        return asd, hd95

    def combined_mean_accuracy(self, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the Combined Mean Accuracy (CMA) based on the provided metrics and coefficients.

        Parameters
        ----------
        metrics : dict[str, torch.Tensor]
            Dictionary containing the computed metrics.

        Returns
        -------
        cma : torch.Tensor
            Combined Mean Accuracy.
        """

        cma = (
            metrics.get("DSC", torch.tensor(0.0)) * self.coef["DSC"]
            + metrics.get("IoU", torch.tensor(0.0)) * self.coef["IoU"]
            + (1 - metrics.get("ASD", torch.tensor(0.0))) * self.coef["ASD"]
            + (1 - metrics.get("HD95", torch.tensor(0.0))) * self.coef["HD95"]
        )
        cma /= sum(self.coef.values())
        return cma


class ClassificationMetrics:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def accuracy(self, pd_cls: torch.Tensor, gt_cls: torch.Tensor) -> torch.Tensor:
        """
        Computes the accuracy of the classification predictions.

        Parameters
        ----------
        pd_cls : torch.Tensor (B,)
            Predicted class labels.

        gt_cls : torch.Tensor (B,)
            Ground truth class labels.

        Returns
        -------
        accuracy : torch.Tensor
            Accuracy of the predictions.
        """

        # (B,) → scalar
        accuracy = (pd_cls == gt_cls).float().mean()
        return accuracy


class Accuracy:
    def __init__(self, cfg: Config):
        """
        Initializes the Accuracy class with the configuration object.

        Parameters
        ----------
        cfg : Config
            Configuration object containing the following attributes:
            - `.DISTANCE_METRICS` (bool): Whether to compute distance metrics.
            - `.CLS_THRESHOLD` (float | None): Threshold for classification.
            - `.WORLD_SIZE` (int): Number of GPUs used for training.
            - `.SEG_CLASSES` (int): Corresponds to the boundary class label in the classification task.
        """
        
        self.sdm_from_mask: bool = cfg.SDM_FROM_MASK
        self.cls_threshold: float | None = cfg.CLS_THRESHOLD
        self.worldsize: int = cfg.WORLD_SIZE
        self.boundary_id: int = cfg.SEG_CLASSES

        self.metrics: dict[str, list[torch.Tensor]] = {}
        self.segMetrics = SegmentationMetrics(cfg)
        self.clsMetrics = ClassificationMetrics(cfg)

    def _logits2predictions(self, 
                            logits: dict[str, torch.Tensor]
                            ) -> tuple[torch.Tensor | None, ...]:
        """
        Converts logits to predictions for classification and segmentation tasks.
        For classification, if the maximum probability for a class exceeds the threshold,
        the class label is assigned; otherwise, the boundary-class label is assigned.
        For segmentation, the predicted 1-hot mask is created by taking the class with the highest logit.

        Parameters
        ----------
        logits : dict[str, torch.Tensor]
            Dictionary containing logits for classification and segmentation.

        Returns
        -------
        pd_cls : torch.Tensor (B,) | None
            Predicted class labels.

        pd_mask : torch.Tensor (B, C, H, W) | None
            Predicted segmentation masks.
        """

        cls_logits = logits.get("cls")
        if cls_logits is not None:
            pd_cls = logits_to_lbl(cls_logits, self.cls_threshold)
        else:
            pd_cls = None

        seg_logits = logits.get("seg")
        if seg_logits is not None:
            pd_mask = logits_to_msk(seg_logits, "1hot")
        else:
            pd_mask = None

        sdm_logits = logits.get("sdm")
        if sdm_logits is not None:
            pd_sdm = torch.tanh(sdm_logits)
        else:
            pd_sdm = None

        return pd_cls, pd_mask, pd_sdm

    def update(self, logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        """
        Update the metrics dictionary with the values for the current batch.

        Parameters
        ----------
        logits : dict[str, torch.Tensor]
            Dictionary containing logits for classification and segmentation.

        batch : dict[str, torch.Tensor]
            Dictionary containing ground truth labels.
        """

        with torch.inference_mode():
            pd_cls, pd_mask, pd_sdm = self._logits2predictions(logits)
            gt_cls, gt_mask, gt_sdm = batch["cls"], batch["mask"], batch["sdm"]

            if pd_cls is not None:
                ttr = self.clsMetrics.accuracy(pd_cls, gt_cls)
                self.metrics.setdefault("TTR", []).append(ttr)

            if pd_mask is not None:
                dsc = self.segMetrics.dice(pd_mask, gt_mask)
                iou = self.segMetrics.iou(pd_mask, gt_mask)
                self.metrics.setdefault("DSC", []).append(dsc)
                self.metrics.setdefault("IoU", []).append(iou)
                
            if pd_mask is not None and pd_sdm is not None:
                valid = (gt_cls == self.boundary_id)
                if valid.any():
                    gt_mask = gt_mask[valid]
                    gt_sdm = gt_sdm[valid]
                    pd_mask = pd_mask[valid]
                    pd_sdm = pd_sdm[valid] if not self.sdm_from_mask else None
                    
                    asd, hd95 = self.segMetrics.boundary(
                        gt_mask, gt_sdm, pd_mask, pd_sdm
                    )
                    self.metrics.setdefault("ASD", []).append(asd)
                    self.metrics.setdefault("HD95", []).append(hd95)

    def compute_avg(self, length: int) -> dict[str, float]:
        """
        Computes the average of the metrics over all batches.

        Parameters
        ----------
        length : int
            Number of batches.

        Returns
        -------
        avgMetrics : dict[str, float]
            Dictionary of averaged metrics.
        """
        avgMetrics = {k: torch.stack(v).mean() for k, v in self.metrics.items()}
        avgMetrics["CMA"] = self.segMetrics.combined_mean_accuracy(avgMetrics)

        if self.worldsize > 1:
            avgMetrics = gather_tensors(avgMetrics, self.worldsize)

        return {k: round(v.item(), 4) for k, v in avgMetrics.items()}

    def reset(self):
        self.metrics: dict[str, list[torch.Tensor]] = {}
