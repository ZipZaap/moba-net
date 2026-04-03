import yaml
import torch
import numpy as np
from pathlib import Path

from utils.dataset import DatasetTools
from utils.util import load_png, save_predictions
from configs.cfgparser import Config
from configs.cli import parse_cli_args
from model.MobaNet import MobaNet


class Predictor:
    def __init__(
        self,
        checkpoint: Path | str | None,
        device: str,
    ):
        """
        Initialize the predictor with the configuration.

        Parameters
        ----------
        checkpoint : Path | str | None
            Path to the model checkpoint.

        device : str
            Device to use for inference (e.g., "cuda:0").

        Raises
        ------
        ValueError
            If the checkpoint path is not specified or does not exist.
        """

        self.device = device

        if (
            checkpoint
            and Path(checkpoint).is_file()
            and Path(checkpoint).suffix == ".pth"
        ):
            # Load model & weights from checkpoint
            _chkp = torch.load(
                checkpoint, map_location=device, mmap=True, weights_only=True
            )
            weights, model_cfg = _chkp["weights"], _chkp["config"]

            # Initialize model with weights, set to eval mode
            self.model = MobaNet(
                model=model_cfg["MODEL"],
                unet_depth=model_cfg["UNET_DEPTH"],
                conv_depth=model_cfg["CONV_DEPTH"],
                in_channels=model_cfg["INPUT_CHANNELS"],
                seg_classes=model_cfg["SEG_CLASSES"],
                cls_classes=model_cfg["CLS_CLASSES"],
                inference=True,
            )

            self.model.load_state_dict(weights)
            self.model = self.model.to(device)
            self.model.eval()
        else:
            raise ValueError("Checkpoint must be an existing file with .pth extension.")

    def predict(
        self, 
        input: str | Path | np.ndarray | torch.Tensor,
        cls_threshold: float | None = None,
        ) -> torch.Tensor:
        """
        Predicts the output for the given input image.

        Parameters
        ----------
        input : str | Path | np.ndarray | torch.Tensor
            Input image path, numpy array, or tensor.
            - np.ndarray: should be of shape (H, W) or (H, W, C) or (B, H, W, C).
            - torch.Tensor: should be of shape (H, W) or (H, W, C) or (B, H, W, C).

        Returns
        -------
        output : torch.Tensor (B, C, H, W)
            The model's output logits tensor.

        Raises
        ------
        ValueError
            If the input array or tensor has an unsupported shape.

        TypeError
            If the input type is unsupported.

        Example
        -------
        >>> from predict import Predictor
        >>> model = Predictor('saved/exp_0/MobaNet-model.pth', 'cuda:0', 0.8)
        >>> imID = 'ESP_123456_1234_RED-1234_1234'
        >>> impath = f'{cfg.IMG_DIR}/{imID}.png'
        >>> output = model.predict(impath)
        """

        # Basic environment check
        if not torch.cuda.is_available():
            raise RuntimeError(
                "This library requires a GPU with CUDA support. "
                "Please verify the PyTorch installation and ensure that a compatible GPU is available."
            )

        if isinstance(input, (str, Path)):
            # Load image from file; (H, W, C)
            input = load_png(input)

            # add batch dimension; (H, W, C) → (1, H, W, C)
            input = input[None, ...]

            # make pytorch compatible; (1, H, W, C) → (1, C, H, W)
            tensor = torch.from_numpy(input).permute(0, 3, 1, 2).float()

        elif isinstance(input, (np.ndarray, torch.Tensor)):
            # Expand dimensions batch and channel dims (if necessary)
            if input.ndim == 2:
                input = input[None, ..., None]  # (H, W) → (1, H, W, 1)
            elif input.ndim == 3:
                input = input[None, ...]  # (H, W, C) → (1, H, W, C)
            elif input.ndim > 4:
                raise ValueError(f"Unsupported input shape: {input.shape}. "
                                 f"Expected an 2D, 3D or 4D (batched) input.")

            # Convert to tensor if it's an np.ndarray and permute dimensions for PyTorch
            tensor = (
                torch.from_numpy(input).float()
                if isinstance(input, np.ndarray)
                else input
            )
            tensor = tensor.permute(0, 3, 1, 2)

        else:
            raise TypeError(f"Unsupported input type: {type(input)}. Expected str, Path, "
                            f"np.ndarray, or torch.Tensor.")

        tensor = tensor.to(self.device)
        with torch.inference_mode():
            return self.model(tensor, cls_threshold)["seg"]

if __name__ == "__main__":
    # Load the YAML into a dict
    with Path("configs/config.yaml").open() as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)

    # Default CLI behavior. Allows: python predict.py --FOO=bar
    cfg = parse_cli_args(cfg, inference=True)

    # Instantiate the Config object
    CONF: Config = Config(cfg, inference=True)

    # Initialize model and data loader
    model = Predictor(CONF.CHECKPOINT, CONF.DEFAULT_DEVICE)
    loader = DatasetTools.predict_dataloader(CONF)

    # Run prediction on batched images
    for batch in loader:
        ids, img_batch = batch["id"], batch["image"].to(CONF.DEFAULT_DEVICE)
        mask_batch = model.predict(img_batch, CONF.CLS_THRESHOLD)
        mask_batch = (mask_batch > CONF.SEG_THRESHOLD)
        save_predictions(CONF.MSK_DIR, mask_batch, ids)
