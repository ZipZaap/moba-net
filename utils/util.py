import cv2
import math
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributed import all_gather


def load_png(impath: str | Path) -> np.ndarray:
    """
    Load a PNG image from the specified path and normalize it to [0, 1].
    If the image is 2D, it is reshaped to 3D with a single channel.
    If the image is 3D, it is loaded in BGR format, which is maintained
    throughout the rest of the processing/training.

    Parameters
    ----------
    impath : str | Path
        Path to the .png image.

    Returns
    -------
    arr : np.ndarray (H, W, C)
        Normalized image array.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.

    ValueError
        If the image shape is not 2D or 3D.
    """

    arr = cv2.imread(str(impath), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Input path {impath} does not exist or is not a file.")

    if arr.ndim == 2:
        arr = arr[..., None]  # shape: (H, W) → (H, W, 1)
    elif arr.ndim > 3:
        raise ValueError(
            f"Unsupported image shape: {arr.shape}. Expected a 2D or 3D image."
        )

    max_val = np.iinfo(arr.dtype).max
    arr = arr.astype(np.float32) / max_val  # Normalize to [0, 1]
    return arr

def load_mask(maskpath: str | Path, seg_classes: int = 1) -> np.ndarray:
    """
    Load a mask from the specified path. If the mask has multiple classes, it is converted to a one-hot encoded format.
    Alternatively, leave the default `seg_classes=1` to skip one-hot encoding and return the mask as is.

    Parameters
    ----------
    maskpath : str | Path
        Path to the mask .png.

    seg_classes : int
        Number of channels in the segmentation mask

    Returns
    -------
    mask : np.ndarray
        - If `seg_classes > 1`:  One-hot encoded mask with shape (H, W, seg_classes).
        - If `seg_classes == 1`: Mask with shape (H, W, 1).

    Raises
    ------
    FileNotFoundError
        If the mask file does not exist.

    ValueError
        If the mask shape is not 2D.
    """

    mask = cv2.imread(str(maskpath), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Input path {maskpath} does not exist or is not a file.")

    if mask.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask.shape}. Expected a 2D mask.")

    if seg_classes > 1:
        return np.eye((seg_classes), dtype=np.uint8)[mask]  # 1Hot; (H, W) → (H, W, C)
    else:
        return mask[..., None]  # (H, W) → (H, W, 1)

def load_sdm(sdmpath: Path, mask_shape: tuple[int, ...]) -> np.ndarray:
    """
    Loads a Signed Distance Map from the specified file path.
    If the given path does not exist, it is assumed that the mask with the
    corresponding ID has no class-to-class boundaries. In such cases, a zero-filled SDM
    with the provided `mask_shape` is returned.

    Parameters
    ----------
    sdmpath : Path
        Path to the SDM .npy file.

    mask_shape : tuple[int, ...] (H, W, C)
        Shape of the mask to create an empty SDM if the file does not exist.

    Returns
    -------
    sdm: np.ndarray (H, W, C)
        - Loaded SDM if file exists.
        - Zeros SDM if file does not exist.

    Raises
    ------
    ValueError
        If `mask_shape` does not conform to (H, W, C).
    """

    if sdmpath.exists():
        return np.load(str(sdmpath))
    else:
        if len(mask_shape) != 3:
            raise ValueError(f"Expected mask_shape to have exactly 3 "
                             f"dimensions (H, W, C); got {mask_shape} instead.")

        return np.zeros((*mask_shape[:-1], 1))

def logits_to_msk(logits, mode: str) -> torch.Tensor:
    """
    Converts models segmentation logits to a segmentation mask based on the specified mode.

    Parameters
    ----------
    logits : torch.Tensor (B, C, H, W) | torch.Tensor (B, 1, H, W)
        Model output logits from segmentation branch.

    mode : str
        Mode for converting logits to mask:
        - `1hot`:    One-hot encoding (X ∈ {0, 1}); non-differentiable.
        - `softmax`: Probability distribution (X ∈ [0; 1]); differentiable.
        - `argmax`:  Pixel-to-class (X ∈ {0, 1, ..., seg_classes}); non-differentiable.

    Returns
    -------
    pd_mask : torch.Tensor
        Segmentation mask after applying the specified mode.
        - If mode is `1hot`, shape: (B, C, H, W)
        - If mode is `softmax`, shape: (B, C, H, W)
        - If mode is `argmax`, shape: (B, 1, H, W)
    """

    if mode == "argmax":
        # 𝑿 ∈ {0, 1, ..., seg_classes}; shape: (B, 1, H, W)
        pd_mask = logits.argmax(dim=1, keepdim=True)
    elif mode == "1hot":
        # 𝑿 ∈ {0, 1}; shape: (B, C, H, W)
        topk = logits.argmax(dim=1, keepdim=True)
        pd_mask = torch.zeros_like(logits).scatter_(1, topk, 1.0)
    else:  # softmax
        # 𝑿 ∈ [0; 1]; shape: (B, C, H, W)
        pd_mask = F.softmax(logits, dim=1)

    return pd_mask

def logits_to_lbl(logits, cls_threshold: float | None) -> torch.Tensor:
    """
    Converts model classification logits to class labels based on the specified threshold.

    Parameters
    ----------
    logits : torch.Tensor (B, C)
        Model output logits from classification branch.

    cls_threshold : float | None
        Threshold for classifying logits into labels. If the maximum probability is 
        below this threshold, the label is set to `C - 1`, i.e. boundary class.
        If `None`, the argmax of the logits is used to determine the class labels.

    Returns
    -------
    pd_cls : torch.Tensor (B,)
        Predicted class labels based on the logits and threshold.
        - If `cls_threshold` is not None, uses the threshold to determine class labels.
        - Otherwise, uses argmax to determine class labels.
    """
        
    B, C = logits.shape

    # Convert logits to probabilities
    cls_probs = F.softmax(logits, dim=1)

    # Get the 'winner' class and its confidence
    max_probs, pd_cls = cls_probs.max(dim=1)

    # Apply thresholding to determine class labels
    if cls_threshold:
        is_confident = max_probs > cls_threshold
        pd_cls = torch.where(is_confident, pd_cls, torch.full_like(pd_cls, C - 1))

    return pd_cls

def test_matching_files(dir1: Path, dir2: Path) -> bool:
    """Checks if two directories contain matching sets of image IDs"""
    files1 = {f.stem for f in dir1.glob("*.png")}
    files2 = {f.stem for f in dir2.glob("*.png")}
    return files1 == files2

def get_rotation_geometry(imdir: Path, input_size: int) -> tuple[int | None, float]:
    """
    Determines whether the images are large enough to accomodate the rotation augmentation
    during training. If the image is too small to perform rotation without upscaling, 
    the function calculates the target size to which the images should be resized. 
    Additionally, the function calculates the margin ratio for the images, which is 
    used to determine the threshold for classifying the images as boundary vs non-boundary.

    Parameters
    ----------
    imdir : Path
        Directory containing the images.

    input_size : int
        Desired input size for the images.

    Returns
    -------
    h_target : int | None
        Target height for the images. None if no resizing is needed.

    margin_ratio : float
        Classification threshold later used in `generate_class_labels`.
        
    Raises
    ------
    ValueError
        If the images are not square.
    """
    
    with Image.open(imdir / next(imdir.glob("*.png"))) as img:
        h, w =int(img.width), int(img.height)
    
    if h != w:
        raise ValueError(f"Expected square images, but got height={h} and width={w}.")
    
    # Nearest even integer that can accommodate the diagonal of the input (rotation)
    h_target = math.ceil(input_size * math.sqrt(2) / 2) * 2
    
    if h >= h_target:
        margin_ratio = (h  + input_size) / (2 * h)
        margin_ratio = math.floor(margin_ratio * 100) / 100
        return None, margin_ratio
    else:
        print(f"[INFO] Image size {h}x{w} is insufficient to perform rotation augment "
                f"without upscaling. Resizing to {h_target}x{h_target} for training.")
        margin_ratio = (h_target  + input_size) / (2 * h_target)
        margin_ratio = math.floor(margin_ratio * 100) / 100
        return h_target, margin_ratio

def resize_and_save(
    all_ids: list[str], 
    h_target: int | None, 
    src_dir: Path,
    dst_dir: Path,
) -> None:
    """Processes the images and saves them to the training directory.

    Parameters
    ----------
    all_ids : list[str]
        List of image IDs to process.
        
    h_target : int | None
        Target height (and width) for resizing. If None, images are not resized.
        
    src_dir : Path
        Directory containing the original images & masks.
        
    dst_dir : Path
        Directory where the processed images & masks will be saved.
    """
    
    src_img_dir = src_dir / 'images'
    src_msk_dir = src_dir / 'masks'
    dst_img_dir = dst_dir / 'images'
    dst_msk_dir = dst_dir / 'masks'
    
    for d in (dst_img_dir, dst_msk_dir):
        d.mkdir(parents=True, exist_ok=True)

    for img_id in tqdm(all_ids, desc="[PREP] Composing dataset"):
        filename = f"{img_id}.png"
        src_img_pth = src_img_dir / filename
        src_msk_pth = src_msk_dir / filename
        dst_img_pth = dst_img_dir / filename
        dst_msk_pth = dst_msk_dir / filename
        
        if not h_target:
            shutil.copy(src_img_pth, dst_img_pth)
            shutil.copy(src_msk_pth, dst_msk_pth)
            continue
        
        img = cv2.imread(str(src_img_pth), cv2.IMREAD_UNCHANGED)
        msk = cv2.imread(str(src_msk_pth), cv2.IMREAD_UNCHANGED)
        
        if img is None or msk is None:
            raise FileNotFoundError(f"Could not read image or mask for ID: {img_id}")
        
        img_resized = cv2.resize(img, (h_target, h_target), 
                                    interpolation=cv2.INTER_LINEAR)
        msk_resized = cv2.resize(msk, (h_target, h_target), 
                                        interpolation=cv2.INTER_NEAREST)
            
        cv2.imwrite(str(dst_img_pth), img_resized)
        cv2.imwrite(str(dst_msk_pth), msk_resized)

def gather_tensors(
    tensor_dict: dict[str, torch.Tensor], 
    worldsize: int
) -> dict[str, torch.Tensor]:
    """
    Gathers tensor values from all GPUs and averages them.

    Parameters
    ----------
    tensor_dict : dict[str, torch.Tensor]
        Dictionary of tensor values (e.g., losses or metrics) per GPU.

    worldsize : int
        Number of processes (GPUs) in the distributed training.

    Returns
    -------
    tensor_dict : dict[str, torch.Tensor]
        Dictionary of averaged tensor values.
    """
    
    for name, tensor in tensor_dict.items():
        avgLst = [torch.zeros_like(tensor) for _ in range(worldsize)]
        all_gather(avgLst, tensor)
        tensor_dict[name] = torch.stack(avgLst).nanmean()
        
    return tensor_dict

def remap_to_sorted_indices(arr: np.ndarray) -> np.ndarray:
    """
    Replace each value in `a` by its index in the sorted list of uniques.
    Example: [0, 7, 100, 245] -> [0, 1, 2, 3]
    """
    _, inv = np.unique(arr, return_inverse=True)
    return inv.reshape(arr.shape)

def save_predictions(maskpath: Path, masks: torch.Tensor, ids: list[str]):
    """
    Save model predictions to the specified directory.

    Parameters
    ----------
    maskpath : Path
        Path to the directory where masks will be saved.

    masks : torch.Tensor (B, C, H, W)
        Model output logits tensor.

    ids : list[str]
        List of image IDs corresponding to the outputs.
    """

    if not maskpath.exists():
        maskpath.mkdir(parents=True)

    for mask, id in zip(masks, ids):
        save_path = str(maskpath / f"{id}.png")
        mask = (mask.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        cv2.imwrite(save_path, mask)

    print(f"[INFO] Predictions saved to {maskpath}")