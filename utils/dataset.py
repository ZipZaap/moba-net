import os
import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Iterator, Sequence

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from configs.cfgparser import Config
from utils.util import load_png, load_mask, load_sdm, \
    test_matching_files, get_rotation_geometry, resize_and_save

class KFold:
    """
    Lightweight replacement for sklearn.model_selection.KFold.
    Balances classes, generates K-Fold train/test splits, and saves to JSON.
    """
    def __init__(
        self, 
        n_splits: int,
        boundary_id: str | int,
        tts_json_path: Path,
        *, 
        shuffle: bool = True, 
        random_state: int = 42,
    ):
        """
        Initialize the KFold object.

        Parameters
        ----------
        n_splits : int
            Number of splits for K-Fold cross-validation.
            
        boundary_id : str | int
            The label ID corresponding to the boundary class.
            
        tts_json_path : Path
            The file path where the resulting JSON splits will be saved.
            
        shuffle : bool
            Whether to shuffle the data before splitting.
            
        random_state : int
            Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.boundary_id = str(boundary_id)
        self.tts_json_path = tts_json_path
        self.shuffle = shuffle
        self.random_state = random_state
        
    def _fetch_ids(
        self,
        D: dict[str, list[str]], 
        idx: list[int]
    ) -> tuple[list[str], list[str]]:
        """
        Splits the dataset into boundary and full images based on the provided indices.

        Parameters
        ----------
        D : dict[str, list[str]]
            Dictionary mapping labels to image IDs.
            
        idx : list[int]
            Indices of the images to be split.

        Returns
        -------
        boundary : list[str]
            List containing boundary image IDs
            
        full : list[str]
            List containing all image IDs
        """
        
        full, boundary = [], []
        for i in idx:
            for lbl, id_list in D.items():
                full.append(id_list[i])
                if lbl == self.boundary_id:
                    boundary.append(id_list[i])

        return boundary, full

    def _split(self, X: Sequence | int) -> Iterator[tuple[list[int], list[int]]]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : Sequence | int
            Data to split; can be a sequence or an integer representing the number of samples.

        Returns
        -------
        Iterator[tuple[list[int], list[int]]]
            An iterator yielding tuples of (train_indices, test_indices) for each fold.
        """
        
        # Accept len(X) or treat X as the sample count
        n_samples = len(X) if not isinstance(X, int) else X

        # Build & optionally shuffle the master index list
        indices = list(range(n_samples))
        if self.shuffle:
            rng = random.Random(self.random_state)
            rng.shuffle(indices)

        # Compute fold sizes (identical to sklearn logic)
        fold_sizes = [n_samples // self.n_splits] * self.n_splits
        for i in range(n_samples % self.n_splits):
            fold_sizes[i] += 1

        # Yield each (train_idx, test_idx) pair
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = indices[:start] + indices[stop:]
            yield train_idx, test_idx
            current = stop

    def create_splits(self, label_to_id: dict[str, list[str]]) -> list[str]:
        """
        Balances classes, generates K-Fold train/test splits, and saves to JSON.

        Parameters
        ----------
        label_to_id : dict[str, list[str]]
            Dictionary mapping class labels to lists of image IDs.

        Returns
        -------
        all_ids : list[str]
            A flattened list of all balanced image IDs used in the splits.
        """
        
        tts_splits = {}
        rng = random.Random(self.random_state)

        # Downsample to balance classes based on the minority class
        n_samples = min(len(samples) for samples in label_to_id.values())
        balanced_labels = {k: rng.sample(v, n_samples) for k, v in label_to_id.items()}
        all_ids = [img_id for samples in balanced_labels.values() for img_id in samples]

        # Generate splits using the balanced sample count (Calling the self.split method)
        for fold, (train_idx, test_idx) in enumerate(self._split(n_samples)):
            
            boundary_train, full_train = self._fetch_ids(balanced_labels, train_idx)
            boundary_test, full_test = self._fetch_ids(balanced_labels, test_idx)

            tts_splits[fold] = {
                "boundary_train": boundary_train,
                "boundary_test": boundary_test,
                "full_train": full_train,
                "full_test": full_test,
            }

        # Save to JSON
        self.tts_json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.tts_json_path.open("w") as f:
            json.dump(tts_splits, f, indent=4)

        return all_ids

# --------------------------------------------------------------------------------------
# PyTorch Dataset objects
# --------------------------------------------------------------------------------------

class FullDataset(Dataset):
    def __init__(self, imIDs: list[str], transforms: A.Compose | None, cfg: Config):
        """
        Initializes the Pytorch Dataset.

        Parameters
        ----------
        imIDs : list[str]
            List of image IDs to be used in the dataset.

        transforms : A.Compose | None
            Albumentations transformations to be applied to the images and masks.

        cfg : Config
            Configuration object with the following attributes:
            - `.IMG_DIR` (Path): Directory containing the input images.
            - `.MSK_DIR` (Path): Directory containing the input masks.
            - `.SDM_DIR` (Path): Directory containing the input SDMs.
            - `.LBL_JSON` (Path): Path to the class labels JSON file.
            - `.CLS_CLASSES` (int): Number of classes for classification.
            - `.SEG_CLASSES` (int): Number of classes for segmentation.
        """

        self.imIDs = imIDs
        self.transforms = transforms

        self.img_dir: Path = cfg.IMG_DIR
        self.msk_dir: Path = cfg.MSK_DIR
        self.sdm_dir: Path = cfg.SDM_DIR
        self.labels: dict = json.load(cfg.LBL_JSON.open())["id_to_label"]
        self.cls_classes: int = cfg.CLS_CLASSES
        self.seg_classes: int = cfg.SEG_CLASSES

    def __len__(self) -> int:
        return len(self.imIDs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        imID = self.imIDs[idx]
        cls = self.labels[imID]

        # load image → (H, W, C)
        impath = self.img_dir / f"{imID}.png"
        image = load_png(impath)

        # load mask → (H, W, C)
        maskpath = self.msk_dir / f"{imID}.png"
        mask = load_mask(maskpath, self.seg_classes)

        # load SDM → (H, W, 1)
        sdmpath = self.sdm_dir / f"{imID}.npy"
        sdm = load_sdm(sdmpath, mask.shape)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask, sdm=sdm)
            image = transformed["image"]
            mask = transformed["mask"]
            sdm = transformed["sdm"]

        # make PyTorch compatible: (H, W, C) → (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        sdm = torch.from_numpy(sdm).permute(2, 0, 1).float()
        cls = torch.tensor(cls).long()

        return {"image": image, "mask": mask, "sdm": sdm, "cls": cls}


class BoundaryMasksDataset(Dataset):
    def __init__(self, imIDs: list[str], msk_dir: Path, seg_classes: int):
        """
        Initializes the BoundaryMasksDataset, which is used when generating SDMs

        Parameters
        ----------
        imIDs : list[str]
            List of image IDs to be used in the dataset.

        msk_dir : Path
            Directory containing the mask images.

        seg_classes : int
            Number of segmentation classes (incl. background).
        """

        self.imIDs = imIDs
        self.msk_dir = msk_dir
        self.seg_classes = seg_classes

    def __len__(self) -> int:
        return len(self.imIDs)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        imID = self.imIDs[idx]

        # load mask → (H, W, C)
        mask = load_mask(self.msk_dir / f"{imID}.png", self.seg_classes)

        # make PyTorch compatible: (H, W, C) → (C, H, W)
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return {"id": imID, "mask": mask}


class PredictDataset(Dataset):
    def __init__(self, impaths: list[Path]):
        """
        Initializes the PredictDataset for inference.

        Parameters
        ----------
        impaths : list[Path]
            List of image paths to be used for prediction.
        """

        self.impaths = impaths

    def __len__(self) -> int:
        return len(self.impaths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        impath = self.impaths[idx]

        # load image → (H, W, C)
        image = load_png(impath)

        # # make PyTorch compatible: (H, W, C) → (C, H, W)
        # image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = torch.from_numpy(image).float()

        return {"id": impath.stem, "image": image}

# --------------------------------------------------------------------------------------
# Dataset management tools
# --------------------------------------------------------------------------------------

class DatasetTools:
    @staticmethod
    def _generate_class_labels(
        msk_dir: Path,
        seg_classes: int,
        threshold: float
    ) -> dict[str, dict[str, list[str]]]:
        """
        Generates class labels for the dataset based on their respective masks.
        If any class label occupies the area greater than `threshold` portion of the mask,
        the image is classified as that class. Otherwise, image is classified as boundary.

        Parameters
        ----------
        msk_dir : Path
            Directory containing the mask images.

        seg_classes : int
            Number of classes in the segmentation task (incl. background). Corresponds to 
            the `id` of the boundary class. Class 1 → 0, class 2 → 1, …, so the (N + 1)-th 
            class is labeled N.

        threshold : float
            Threshold value to classify the images.

        Returns
        -------
        lbl_dict : dict[str, dict[str, list[str]]]
            A dictionary containing image IDs organized by class label & boundary status.
        """

        imIDs = [p.stem for p in msk_dir.glob("*.png")]
        label_to_id, id_to_label = {}, {}

        for id in tqdm(imIDs, desc="[PREP] Generating class labels"):
            maskpath = msk_dir / f"{id}.png"
            mask = load_mask(maskpath)

            counts = np.bincount(mask.ravel(), minlength=seg_classes)
            max_label = int(counts.argmax())
            max_fraction = counts[max_label] / mask.size

            lbl = max_label if max_fraction >= threshold else seg_classes

            label_to_id.setdefault(str(lbl), []).append(id)
            id_to_label[id] = lbl

        return {"label_to_id": label_to_id, "id_to_label": id_to_label}
    
    @classmethod
    def compose_dataset(cls, cfg: Config):
        """
        Generate the class labels and the train/test split files. Maintains 1:1 ratio of 
        all classes in the full dataset.Upscales the images if they are too small to 
        accommodate the rotation augmentation during training.

        Parameters
        ----------
        cfg : Config
            Configuration object with following attributes:
            - `.DATASET_DIR` (Path): Directory containing the original dataset.
            - `.TRAIN_DIR` (Path): Directory to save the composed training dataset.
            - `.BASE_IMG_DIR` (Path): Directory containing the original images.
            - `.BASE_MSK_DIR` (Path): Directory containing the original masks.
            - `.LBL_JSON` (Path): Path to the class labels JSON file.
            - `.TTS_JSON` (Path): Path to save the train/test splits JSON file.
            - `.INPUT_SIZE` (int): Desired input size for the images.
            - `.SEG_CLASSES` (int): Number of segmentation classes (incl. background).
            - `.NUM_KFOLDS` (int): Number of folds for K-Fold cross-validation.
            - `.RANDOM_SEED` (int): Random seed for reproducibility.
            
        Raises
        ------
        ValueError
            If the image and mask directories do not contain matching sets of image IDs.
        """
        
        if not test_matching_files(cfg.BASE_IMG_DIR, cfg.BASE_MSK_DIR):
            raise ValueError(
                f"Image and mask files do not match. {cfg.BASE_IMG_DIR} and "
                f"{cfg.BASE_MSK_DIR} must contain identical sets of image IDs."
            )
            
        h_target, threshold = get_rotation_geometry(cfg.BASE_IMG_DIR, cfg.INPUT_SIZE)
    
        if cfg.LBL_JSON.exists():
            with cfg.LBL_JSON.open() as f:
                lbl_dict = json.load(f)       
        else:
            lbl_dict = cls._generate_class_labels(cfg.BASE_MSK_DIR, cfg.SEG_CLASSES, threshold)
            with cfg.LBL_JSON.open("w") as f:
                json.dump(lbl_dict, f)
                
                
        if cfg.TRAIN_DIR.exists() and any(cfg.TRAIN_DIR.iterdir()):
            print("[INFO] Train directory already exists and is not empty. "
                  "Assuming dataset is already composed. To re-generate the dataset, "
                  "and update the train/test split please clear the train directory.")

        else:
            kf = KFold(
                n_splits=cfg.NUM_KFOLDS,
                boundary_id=cfg.SEG_CLASSES,
                tts_json_path=cfg.TTS_JSON,
                random_state=cfg.SEED,
            )
            all_ids = kf.create_splits(lbl_dict["label_to_id"])
            resize_and_save(all_ids, h_target, cfg.DATASET_DIR, cfg.TRAIN_DIR)

    @classmethod
    def train_dataloaders(cls, cfg: Config) -> tuple[DataLoader, DataLoader]:
        """
        Returns the train and test dataloaders for the specified fold.

        Parameters
        ----------
        cfg : Config
            Configuration object with the following attributes:
            - `.RANK` (int): Rank of the current process.
            - `.FOLD` (int): Fold number for K-Fold cross-validation.
            - `.WORLD_SIZE` (int): Number of GPUs used for distributed training.
            - `.BATCH_SIZE` (int): Batch size for the dataloaders.
            - `.NUM_WORKERS` (int): Number of workers for the dataloaders.
            - `.TTS_JSON` (Path): Path to the train/test splits JSON file.
            - `.TRAIN_SET` (str): Train set composition.
            - `.TEST_SET` (str): Test set composition.
            - `.INPUT_SIZE` (int): Size to which the images will be resized/cropped.
            
        Returns
        -------
        trainLoader, testLoader : tuple
            Train and test dataloaders.
        """

        with cfg.TTS_JSON.open() as f:
            TTS = json.load(f)

        trainIDs = TTS[str(cfg.DEFAULT_FOLD)][f"{cfg.TRAIN_SET}_train"]
        testIDs = TTS[str(cfg.DEFAULT_FOLD)][f"{cfg.TEST_SET}_test"]
        
        if cfg.RANK == 0:
            print(
                f"[INFO] Samples (Total/Train/Test): "
                f"{len(trainIDs) + len(testIDs)} // {len(trainIDs)} // {len(testIDs)}"
            )

        train_transform = A.Compose([
            # --- GEOMETRIC TRANSFORMATIONS ---
            # 1. Orientation Invariance
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ],p=0.5,),
            
            # 2. Rotation (+1s / 700 images)
            A.Rotate(limit=180, p=1.0, border_mode=0),
            
            # 3. Final Sizing
            A.CenterCrop(height=cfg.INPUT_SIZE, width=cfg.INPUT_SIZE, p=1.0),
            
            # --- PHOTOMETRIC TRANSFORMATIONS ---
            # 4. Sensor & Atmosphere Simulation
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05), per_channel=False, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ],p=0.2,),
            
            # 5. Contrast & Brightness (+1s / 700 images)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ],p=0.5,),
        ], additional_targets={"mask": "mask", "sdm": "mask"},)
        
        test_transform = A.Compose([
            A.CenterCrop(height=cfg.INPUT_SIZE, width=cfg.INPUT_SIZE, p=1.0),
        ], additional_targets={"mask": "mask", "sdm": "mask"},)
        
        trainSet = FullDataset(imIDs=trainIDs, transforms=train_transform, cfg=cfg)
        testSet = FullDataset(imIDs=testIDs, transforms=test_transform, cfg=cfg)

        trainSampler = (
            DistributedSampler(
                trainSet,
                num_replicas=cfg.WORLD_SIZE,
                rank=cfg.RANK,
                shuffle=True,
                drop_last=True,
            )
            if cfg.WORLD_SIZE > 1
            else None
        )

        testSampler = (
            DistributedSampler(
                testSet,
                num_replicas=cfg.WORLD_SIZE,
                rank=cfg.RANK,
                shuffle=False,
                drop_last=True,
            )
            if cfg.WORLD_SIZE > 1
            else None
        )

        trainLoader = DataLoader(
            trainSet,
            batch_size=cfg.BATCH_SIZE,
            pin_memory=torch.cuda.is_available(),
            shuffle=(cfg.WORLD_SIZE <= 1),
            sampler=trainSampler,
            num_workers=cfg.NUM_WORKERS,
            persistent_workers=cfg.NUM_WORKERS > 0,
        )

        testLoader = DataLoader(
            testSet,
            batch_size=cfg.BATCH_SIZE,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            sampler=testSampler,
            num_workers=cfg.NUM_WORKERS,
            persistent_workers=cfg.NUM_WORKERS > 0,
        )

        return trainLoader, testLoader

    @classmethod
    def predict_dataloader(cls, cfg: Config) -> DataLoader:
        """
        Returns the dataloader used for inference.

        Parameters
        ----------
        cfg : Config
            Configuration object with the following attributes:
            - `.IMG_DIR` (Path): Directory containing the images for inference.
            - `.BATCH_SIZE` (int): Batch size for the dataloader.
            - `.NUM_WORKERS` (int): Number of workers for the dataloader.

        Returns
        -------
        predictLoader : DataLoader
            Dataloader for the images in the inference dataset.
        """

        impaths = [impath for impath in cfg.IMG_DIR.glob("*.png")]
        predictLoader = DataLoader(
            PredictDataset(impaths),
            batch_size=cfg.BATCH_SIZE,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            persistent_workers=cfg.NUM_WORKERS > 0,
        )

        return predictLoader

    @classmethod
    def boundary_mask_dataloader(cls, cfg: Config) -> DataLoader:
        """
        Returns the dataloader used to retrieve masks when generating the SDMs dataset.

        Parameters
        ----------
        cfg : Config
            Configuration object with the following attributes:
            - `.MSK_DIR` (Path): Directory containing the masks.
            - `.LBL_JSON` (Path): Path to the class labels JSON file.
            - `.BATCH_SIZE` (int): Batch size for the dataloader.
            - `.NUM_WORKERS` (int): Number of workers for the dataloader.
            - `.SEG_CLASSES` (int): Number of segmentation classes.
        """

        with cfg.LBL_JSON.open() as f:
            imIDs = json.load(f)["label_to_id"][str(cfg.SEG_CLASSES)]

        loader = DataLoader(
            BoundaryMasksDataset(imIDs, cfg.MSK_DIR, cfg.SEG_CLASSES),
            batch_size=cfg.BATCH_SIZE,
            pin_memory=torch.cuda.is_available(),
            num_workers=cfg.NUM_WORKERS,
            persistent_workers=(cfg.NUM_WORKERS > 0),
        )

        return loader