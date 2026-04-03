import math
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.dataset import DatasetTools
from configs.cfgparser import Config


class SDF:
    @classmethod
    def _generate_kernel(
        cls, 
        device: torch.device, 
        distance: str, 
        K: int,
    ) -> torch.Tensor:
        """Generates a kernel for the Signed Distance Transform approximation.

        Parameters
        ----------
        device : str
            Device to create the kernel on.

        distance : str
            Distance metric used for the SDM ('euclidean', 'manhattan', 'chebyshev').

        K : int
            Size of the kernel. Must be odd.

        Returns
        -------
            kernel : torch.Tensor (1, 1, K, K)
                Kernel tensor.
        """

        center = K // 2
        y, x = torch.meshgrid(
            torch.arange(K, dtype=torch.float32, device=device),
            torch.arange(K, dtype=torch.float32, device=device),
            indexing="ij",
        )

        if distance == "manhattan":
            kernel = torch.abs(x - center) + torch.abs(y - center)

        elif distance == "chebyshev":
            kernel = torch.max(torch.abs(x - center), torch.abs(y - center))

        else:  # 'euclidean'
            kernel = torch.sqrt((x - center) ** 2 + (y - center) ** 2)

        return kernel.view(1, 1, K, K)

    @classmethod
    def _normalize_sdm(
        cls, 
        sdm: torch.Tensor, 
        mask: torch.Tensor, 
        normalization: str,
    ) -> torch.Tensor:
        """
        Normalizes the Signed Distance Map (SDM) based on the specified normalization methods:
        - `dynamic_max`: Normalize by the maximum distance value of each individual sdm.
        - `minmax`: Normalize by both maximum and minimum distance values of each individual sdm.

        Parameters
        ----------
        sdm : torch.Tensor (B, 1, H, W)
            Input Signed Distance Map tensor.

        mask : torch.Tensor (B, C, H, W)
            Input mask tensor.

        normalization : str
            Normalization method (`dynamic_max`, `minmax`).

        Returns
        -------
        sdm : torch.Tensor (B, 1, H, W)
            Normalized Signed Distance Map tensor.
        """

        B = sdm.shape[0]
        mask = mask.sum(dim=(1), keepdim=True)

        if normalization == "dynamic_max":
            maxval = sdm.view(B, -1).max(dim=1)[0]
            sign_mask = torch.where(mask == 0, -1, 1)
            return sdm / (maxval.view(B, 1, 1, 1) * sign_mask)

        else:  # minmax
            minval = torch.multiply(sdm, mask == 1)
            minval = minval.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
            minval = torch.multiply(mask == 1, minval)

            maxval = torch.multiply(sdm, mask == 0)
            maxval = maxval.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
            maxval = torch.multiply(mask == 0, maxval)

            return sdm / (minval - maxval)

    @classmethod
    def compute_sobel_edges(
        cls, 
        mask: torch.Tensor, 
        collapse: bool = True
    ) -> torch.Tensor:
        """Computes the Sobel edges of the input mask.

        If the input mask has multiple channels/classes, it computes the Sobel edges
        for each channel separately and then sums them up to create a single edge map.

        Parameters
        ----------
        mask : torch.Tensor (B, C, H, W)
            Input mask tensor.

        Returns
        -------
        sobel_edges : torch.Tensor (B, 1, H, W)
            Sobel edges tensor.

        Raises
        ------
        ValueError
            If the input mask does not have the expected shape of (B, C, H, W).
        """

        if mask.ndim != 4:
            raise ValueError("Mask must have shape (B, C, H, W).")

        C: int = mask.shape[1]
        device, dtype = mask.device, mask.dtype

        sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device
            )
            .view(1, 1, 3, 3)
            .repeat(C, 1, 1, 1)
        )
        sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device
            )
            .view(1, 1, 3, 3)
            .repeat(C, 1, 1, 1)
        )

        # depth‑wise convolution (one kernel per channel)
        padded = F.pad(mask, (1, 1, 1, 1), mode="reflect")
        grad_x = F.conv2d(padded, sobel_x, groups=C)  # (B, C, H, W)
        grad_y = F.conv2d(padded, sobel_y, groups=C)  # (B, C, H, W)

        # per‑channel gradient magnitude
        per_class_edges = torch.sqrt(grad_x**2 + grad_y**2)  # (B, C, H, W)

        # collapse channels → single edge map
        if collapse:
            sobel_edges = per_class_edges.sum(dim=1, keepdim=True)  # (B, 1, H, W)
            return (sobel_edges > 0).to(dtype)
        else:
            return (per_class_edges > 0).to(dtype)  # (B, C, H, W)

        #  sharpening to [0, 1] - diffrentiable option (training)
        # sobel_edges = torch.sigmoid((sobel_edges - 3) * 100)

        #  sharpening to [0, 1] - non-diffrentiable option (inference)
        # sobel_edges =  (sobel_edges > 0).to(dtype)

    @classmethod
    def sdf(
        cls,
        mask: torch.Tensor,
        K: int,
        distance: str = "chebyshev",
        normalization: str | None = "minmax",
        h: float = 0.35,
    ) -> torch.Tensor:
        """Computes the Signed Distance Map (SDM) using the cascaded convolution method.

        See Pham et. al (https://doi.org/10.1007/978-3-030-71278-5_31) for more details.
        Modified from (https://github.com/kornia/kornia/blob/main/kornia/contrib/distance_transform.py);
        Copyright 2018 Kornia Team; Licensed under the Apache License, Version 2.0.

        In multichannel case, the edge map is computed for each channel separately, and the summed to create
        a single (binary) edge map. This unified map is then used to compute the SDM, i.e. only one calculation
        is performed regardless of the number of channels/classes.

        Parameters
        ----------
        mask : torch.Tensor (B, C, H, W)
            Input mask tensor.

        K : int
            Size of the kernel. Must be odd.

        imsize : int
            Size of the input image.

        distance : str
            Distance metric to use for the SDF.

        normalization : str
            Normalization method to use for the SDF.

        h : float
            Parameter for the exponential kernel.

        Returns
        -------
        sdm : torch.Tensor (B, 1, H, W)
            Signed Distance Map tensor.
        """

        edges = cls.compute_sobel_edges(mask)
        n_iters = math.ceil(max(edges.shape[2], edges.shape[3]) / math.floor(K / 2))
        kernel = cls._generate_kernel(mask.device, distance, K)
        kernel = torch.exp(kernel / -h)
        sdm = torch.zeros_like(edges)

        boundary = edges.clone()
        for i in range(n_iters):
            cdt = F.conv2d(boundary, kernel, padding=(K // 2))
            cdt = -h * torch.log(cdt)
            cdt = torch.nan_to_num(cdt, posinf=0.0)

            edges = torch.where(cdt > 0, 1.0, 0.0)
            if edges.sum() == 0:
                break

            offset = i * (K // 2)
            sdm += (offset + cdt) * edges
            boundary += edges

        if normalization:
            sdm = cls._normalize_sdm(sdm, mask, normalization)

        return sdm

    @classmethod
    def generate_sdms(
        cls, 
        cfg: Config,
        overwrite: bool = False
    ):
        """Generate signed-distance maps (SDMs) for a dataset.

        The routine works with any number of **non-overlapping classes**, producing one
        SDM for every input image/mask. Masks are streamed in mini-batches through a
        PyTorch `DataLoader`. SDMs are saved as **float32** NumPy arrays with shape
        `(H, W, 1)` (channels-last) in `cfg.SDM_DIR`. Filenames mirror those of the
        corresponding masks, e.g. `masks/id_012.png  →  sdms/id_012.npy`.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite existing SDMs.

        cfg : Config
            Configuration object must expose:
            - .SDM_DIR (Path): Directory to save the SDMs.
            - .DEFAULT_DEVICE (str): Device to run the SDM generation on.
            - .SDM_KERNEL_SIZE (int): Size of the kernel. Must be odd.
            - .INPUT_SIZE (int): Size of the input image.
            - .SDM_DISTANCE (str): Distance metric to use for the SDF.
            - .SDM_NORMALIZATION (str): Normalization method to use for the SDF.

        Example
        -------
        >>> from utils.sdf import SDF
        >>> from configs.cfgparser import Config
        >>> cfg = Config('configs/config.yaml', cli=False)
        >>> SDF.generate_sdms(cfg, overwrite=True)
        """

        sdm_dir: Path = cfg.SDM_DIR
        device: str = cfg.DEFAULT_DEVICE
        K: int = cfg.SDM_KERNEL_SIZE
        distance: str = cfg.SDM_DISTANCE
        normalization: str = cfg.SDM_NORMALIZATION

        if not sdm_dir.exists():
            sdm_dir.mkdir(exist_ok=True)

        if not any(sdm_dir.iterdir()) or overwrite:
            loader = DatasetTools.boundary_mask_dataloader(cfg)
            for batch in tqdm(loader, desc=f"[PREP] Generating SDMs on {device}"):
                imIDs = batch["id"]
                masks = batch["mask"].to(device)
                sdms = SDF.sdf(masks, K, distance, normalization)

                for id, sdm in zip(imIDs, sdms):
                    sdm = sdm.permute(1, 2, 0)  # (1, H, W) → (H, W, 1) before saving
                    sdm_np = sdm.detach().cpu().numpy()
                    np.save(str(sdm_dir / f"{id}.npy"), sdm_np)
        else:
            print(f"[INFO] Using cached SDMs. To update the SDMs, delete {sdm_dir} "
                  f"or set `overwrite=True`.")