from pathlib import Path
from types import SimpleNamespace


class Validator:
    type_mapping = {
        "int": int,
        "float": float,
        "str": str,
        "Path": str,
        "bool": bool,
        "list": list,
        "dict": dict,
    }
    

    @classmethod
    def validate_cfg(cls, cfg_dict: dict[str, dict], inference: bool):
        """
        Validates the configuration parameters specified in `cfg_dict`.

        Parameters
        ----------
        cfg_dict : dict[str, dict]
            Dictionary containing configuration parameters and their values.

        inference : bool
            If True, the configuration is for inference mode.

        Raises
        ------
        TypeError
            If the type of a parameter does not match the expected type.

        ValueError
            If the value of a parameter fails the built-in tests.
        """

        # simplify the cfg_dict to a SimpleNamespace object, keeping only the default values
        cls.cfg = SimpleNamespace(
            **{outer: inner["default"] for outer, inner in cfg_dict.items()}
        )
        cls.inference = inference

        for name, parameter in cfg_dict.items():
            value = parameter.get("default")
            options = parameter.get("choices", [])
            expected_type = cls.type_mapping[parameter["type"]]

            # Check if the value is of the expected type
            if value is not None and not isinstance(value, expected_type):
                raise TypeError(
                    f"Expected type `{expected_type.__name__}` for parameter `{name}`, "
                    f"but got type `{type(value).__name__}` instead."
                )

            # Apply built-in validation tests
            attr_name = f"_validate_{name.lower()}"
            if hasattr(cls, attr_name):
                getattr(cls, attr_name)(value, options)
            else:
                print(f"[WARN] No validation test implemented for `{name}`.")

        print("[INFO] Configuration file passed all validation tests.")

    # --- DIRECTORIES ---
    @classmethod
    def _validate_dataset_dir(cls, value, options):
        dataset_dir = Path(value)
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise ValueError(
                f"`DATASET_DIR` : {dataset_dir} does not exist or is not a directory."
            )

        if cls.inference:
            images_dir = dataset_dir / "predict" / "images"
            if not images_dir.exists() or not images_dir.is_dir():
                raise ValueError(
                    "`DATASET_DIR` must contain a subfolder 'predict/images' for inference."
                )
            if not any(images_dir.glob("*.png")):
                raise ValueError(
                    f"`IMG_DIR` : {images_dir} subfolder does not contain any .png image files to run inference on."
                )

        else:
            images_dir = dataset_dir / "images"
            if not images_dir.exists() or not images_dir.is_dir():
                raise ValueError(
                    "`DATASET_DIR` must contain an 'images' subfolder."
                )
            if not any(images_dir.glob("*.png")):
                raise ValueError(
                    f"`IMG_DIR` : {images_dir} subfolder does not contain any .png image files to run training on."
                )

            masks_dir = dataset_dir / "masks"
            if not masks_dir.exists() or not masks_dir.is_dir():
                raise ValueError(
                    "`DATASET_DIR` must contain a 'masks' subfolder."
                )
            if not any(masks_dir.glob("*.png")):
                raise ValueError(
                    f"`MSK_DIR` : {masks_dir} subfolder does not contain any .png mask files to run training on."
                )

    @classmethod
    def _validate_results_dir(cls, value, options):
        pass

    # --- DATASET PARAMETERS ---
    @classmethod
    def _validate_seed(cls, value, options):
        pass

    @classmethod
    def _validate_train_set(cls, value, options):
        if value not in options:
            raise ValueError(f"Value of `TRAIN_SET` must be one of {options}.")

        if (
            cls.cfg.MODEL in ["MobaNet_C", "MobaNet_EC", "MobaNet_EDC"]
            and value == "boundary"
        ):
            raise ValueError(
                "Value conflict between `MODEL` and `TRAIN_SET`! "
                f"Model `{cls.cfg.MODEL}` cannot be trained with `boundary` dataset."
            )

    @classmethod
    def _validate_test_set(cls, value, options):
        if value not in options:
            raise ValueError(f"Value of `TEST_SET` must be one of {options}.")

        if (
            cls.cfg.MODEL in ["MobaNet_C", "MobaNet_EC", "MobaNet_EDC"]
            and value == "boundary"
        ):
            raise ValueError(
                "Value conflict between `MODEL` and `TEST_SET`! "
                f"Model `{cls.cfg.MODEL}` cannot be tested on `boundary` dataset."
            )

    @classmethod
    def _validate_test_split(cls, value, options):
        if not 0 < value < 1:
            raise ValueError("Value of `TEST_SPLIT` must be between 0 and 1.")

    @classmethod
    def _validate_cross_validation(cls, value, options):
        pass

    @classmethod
    def _validate_default_fold(cls, value, options):
        if value < 0:
            raise ValueError("Value of `DEFAULT_FOLD` cannot be negative.")

        if value > int(1 / cls.cfg.TEST_SPLIT):
            raise ValueError(
                "Value conflict between `DEFAULT_FOLD` and `TEST_SPLIT`! "
                "Value of `DEFAULT_FOLD` must be less than 1/`TEST_SPLIT`."
            )

    @classmethod
    def _validate_num_workers(cls, value, options):
        if value < 0:
            raise ValueError("Value of `NUM_WORKERS` cannot be negative.")

    # --- MODEL CONFIGURATION ---
    @classmethod
    def _validate_model(cls, value, options):
        if value not in options:
            raise ValueError(f"Value of `MODEL` must be one of {options}.")

    @classmethod
    def _validate_checkpoint(cls, value, options):
        if value is None and cls.cfg.MODEL in ["MobaNet_C", "MobaNet_D"]:
            raise ValueError(
                "Value conflict between `MODEL` and `CHECKPOINT`! "
                f"Model `{cls.cfg.MODEL}` requires `CHECKPOINT` to be specified."
            )

        if value is None and cls.inference:
            raise ValueError("Inference mode requires `CHECKPOINT` to be specified.")

        if value is not None and (
            not Path(value).is_file() or Path(value).suffix != ".pth"
        ):
            raise ValueError(
                f"`CHECKPOINT`: {value} must be an existing file with .pth extension."
            )

    @classmethod
    def _validate_input_size(cls, value, options):
        if value <= 0:
            raise ValueError("Value of `INPUT_SIZE` must be greater than 0.")

        if value % 2 != 0:
            raise ValueError("Value of `INPUT_SIZE` must be an even number.")

        # review this test later
        if value / (2 ** (cls.cfg.UNET_DEPTH - 1)) < 4:
            raise ValueError(
                "Value conflict between `INPUT_SIZE` and `UNET_DEPTH`! "
                "`INPUT_SIZE/2^(UNET_DEPTH - 1)` must be greater than or equal to 4."
            )

    @classmethod
    def _validate_input_channels(cls, value, options):
        if value not in options:
            raise ValueError(f"Value of `INPUT_CHANNELS` must be one of {options}.")

    @classmethod
    def _validate_unet_depth(cls, value, options):
        if value < 2:
            raise ValueError(
                "Value of `UNET_DEPTH` must be greater than or equal to 2."
            )

        # review this test later
        if cls.cfg.INPUT_SIZE / (2 ** (value - 1)) < 4:
            raise ValueError(
                "Value conflict between `INPUT_SIZE` and `UNET_DEPTH`! "
                "`INPUT_SIZE/2^(UNET_DEPTH - 1)` must be greater than or equal to 4."
            )

    @classmethod
    def _validate_conv_depth(cls, value, options):
        if value <= 0:
            raise ValueError("Value of `CONV_DEPTH` must be greater than 0.")

    @classmethod
    def _validate_batch_size(cls, value, options):
        if value <= 0:
            raise ValueError("Value of `BATCH_SIZE` must be greater than 0.")

    # --- TRAINING OBJECTIVES ---
    @classmethod
    def _validate_loss(cls, value, options):
        loss_set = set(value)
        seg_losses = {"SoftDICE", "HardDICE", "IoU", "SegCE", "wSegCE"}
        sdm_losses = {"wSegCE", "MAE", "sMAE", "cMAE", "Boundary"}
        cls_losses = {"ClsCE"}

        # basic checks
        if not loss_set.issubset(set(options)):
            raise ValueError(
                f"Value of `LOSS` must either be one of or a combination of {options}."
            )
        if len(value) != len(loss_set):
            raise ValueError(
                "Value of `LOSS` must not contain duplicate loss functions."
            )

        # model-specific constraints
        if cls.cfg.MODEL in ["MobaNet_EDC", "MobaNet_ED", "MobaNet_D", "UNet"]:
            if not seg_losses & loss_set:
                raise ValueError(
                    "Value conflict between `MODEL` and `LOSS`! "
                    f"Model `{cls.cfg.MODEL}` requires a segmentation loss to be included in the loss term."
                )
        if cls.cfg.MODEL in ["MobaNet_EDC", "MobaNet_C", "MobaNet_EC"]:
            if not cls_losses & loss_set:
                raise ValueError(
                    "Value conflict between `MODEL` and `LOSS`! "
                    f"Model `{cls.cfg.MODEL}` requires a classification loss to be included in the loss term."
                )
        if cls.cfg.MODEL in ["MobaNet_C", "MobaNet_EC"]:
            if sdm_losses & loss_set:
                raise ValueError(
                    "Value conflict between `MODEL` and `LOSS`! "
                    f"Model `{cls.cfg.MODEL}` cannot be trained with an SDM-based loss."
                )
        if cls.cfg.MODEL in ["MobaNet_ED", "MobaNet_D", "UNet"]:
            if cls_losses & loss_set:
                raise ValueError(
                    "Value conflict between `MODEL` and `LOSS`! "
                    f"Model `{cls.cfg.MODEL}` cannot be trained with a classification loss."
                )

        # dataset constraints
        if cls_losses & loss_set:
            if cls.cfg.TRAIN_SET == "boundary":
                raise ValueError(
                    "Value conflict between `LOSS` and `TRAIN_SET`! "
                    "ClsCE loss cannot be trained with `boundary` dataset, use `full` dataset instead."
                )
            if cls.cfg.TEST_SET == "boundary":
                raise ValueError(
                    "Value conflict between `LOSS` and `TEST_SET`! "
                    "ClsCE loss cannot be tested with `boundary` dataset, use `full` dataset instead."
                )
                
        if not sdm_losses & loss_set and cls.cfg.SDM_FROM_MASK:
            raise ValueError(
                "Value conflict between `LOSS` and `SDM_FROM_MASK`! "
                "If `SDM_FROM_MASK == True`, at least one SDM-based loss must be included in the loss term."
            )
    
    @classmethod
    def _validate_clamp_delta(cls, value, options):
        if not 0 < value < 1:
            raise ValueError("Value of `CLAMP_DELTA` must be between 0 and 1.")

    @classmethod
    def _validate_static_weights(cls, value, options):
        if len(cls.cfg.LOSS) > 1:
            if value:
                if len(value) != len(cls.cfg.LOSS):
                    raise ValueError(
                        "`STATIC_WEIGHTS` must be the same length as the number of loss functions in `LOSS`."
                    )
            else:
                print(
                    "[WARN] Training with multiple losses & `STATIC_WEIGHTS == null`; "
                    "Adaptive loss weighting will be used."
                )

    @classmethod
    def _validate_seg_classes(cls, value, options):
        if value < 2:
            raise ValueError(
                "Value of `SEG_CLASSES` must be greater than or equal to 2, i.e. at least 1 foreground and 1 background class."
            )

    @classmethod
    def _validate_seg_dropout(cls, value, options):
        if value is None or not 0 <= value <= 1:
            raise ValueError("Value of `SEG_DROPOUT` must be between 0 and 1.")

    @classmethod
    def _validate_cls_classes(cls, value, options):
        if value < 3:
            raise ValueError(
                "Value of `CLS_CLASSES` must be greater than or equal to 3, i.e. given N `SEG_CLASSES`, "
                "N+1 classes required to denote images with class-to-class boundaries."
            )

    @classmethod
    def _validate_cls_dropout(cls, value, options):
        if value is None or not 0 <= value <= 1:
            raise ValueError("Value of `CLS_DROPOUT` must be between 0 and 1.")

    @classmethod
    def _validate_cls_threshold(cls, value, options):
        if value is not None and not value > 0:
            raise ValueError(
                "Value of `CLS_THRESHOLD` must either be `null` for no thresholding or greater than 0."
            )

    # --- OPTIMIZER SETTINGS ---
    @classmethod
    def _validate_init_lr(cls, value, options):
        if not 0 < value < 1:
            raise ValueError("Value of `INIT_LR` must be between 0 and 1.")

        if value >= cls.cfg.BASE_LR:
            raise ValueError(
                "Value conflict between `INIT_LR` and `BASE_LR`! "
                "`INIT_LR` must be less than `BASE_LR`."
            )

    @classmethod
    def _validate_base_lr(cls, value, options):
        if not 0 < value < 1:
            raise ValueError("Value of `BASE_LR` must be between 0 and 1.")

        if value <= cls.cfg.INIT_LR:
            raise ValueError(
                "Value conflict between `BASE_LR` and `INIT_LR`! "
                "`BASE_LR` must be greater than `INIT_LR`."
            )

    @classmethod
    def _validate_l2_decay(cls, value, options):
        if not 0 < value < 1:
            raise ValueError("Value of `L2_DECAY` must be between 0 and 1.")

    @classmethod
    def _validate_warmup_epochs(cls, value, options):
        if value < 0:
            raise ValueError("Value of `WARMUP_EPOCHS` cannot be negative.")

    @classmethod
    def _validate_train_epochs(cls, value, options):
        if value <= 0:
            raise ValueError("Value of `TRAIN_EPOCHS` must be greater than 0.")

    # --- SDM PARAMETERS ---
    @classmethod
    def _validate_sdm_kernel_size(cls, value, options):
        if value <= 0:
            raise ValueError("Value of `SDM_KERNEL` must be greater than 0.")

        if value % 2 != 1:
            raise ValueError("Value of `SDM_KERNEL` must be an odd number.")

    @classmethod
    def _validate_sdm_distance(cls, value, options):
        if value not in options:
            raise ValueError(f"Value of `SDM_KERNEL_TYPE` must be one of {options}.")

    @classmethod
    def _validate_sdm_normalization(cls, value, options):
        if value not in options:
            raise ValueError(f"Value of `SDM_NORMALIZATION` must be one of {options}.")

    # --- EVALUATION METRICS ---    
    @classmethod
    def _validate_checkpoint_interval(cls, value, options):
        if value is not None and value <= 0:
            raise ValueError("Value of `CHECKPOINT_INTERVAL` must be greater than 0.")

        if value is not None and value > cls.cfg.TRAIN_EPOCHS:
            raise ValueError(
                "Value conflict between `CHECKPOINT_INTERVAL` and `TRAIN_EPOCHS`! "
                "`CHECKPOINT_INTERVAL` must be less than or equal to `TRAIN_EPOCHS`."
            )

    @classmethod
    def _validate_eval_metric(cls, value, options):
        if value not in options:
            raise ValueError(f"Value of `EVAL_METRIC` must be one of {options}.")

    @classmethod
    def _validate_cma_coefficients(cls, value, options):
        if sum(v != 0 for v in value.values()) < 2:
            raise ValueError(
                "`CMA_COEFFICIENTS` must contain at least 2 non-zero coefficients."
            )

        cma_distances = [k for k in ("ASD", "HD95") if value[k] != 0]
        if not cls.cfg.DISTANCE_METRICS and cma_distances:
            raise ValueError(
                f"`CMA_COEFFICIENTS` includes non-zero coefficient(s) associated with {cma_distances} distance metric(s), "
                "but none of the distance metrics are calculated due to `DISTANCE_METRICS == False`."
            )

    @classmethod
    def _validate_sdm_from_mask(cls, value, options):
        pass

    # --- DDP SETTINGS ---
    @classmethod
    def _validate_gpus(cls, value, options):
        pass

    @classmethod
    def _validate_master_addr(cls, value, options):
        pass

    @classmethod
    def _validate_master_port(cls, value, options):
        pass

    @classmethod
    def _validate_nccl_p2p(cls, value, options):
        pass

    # -- LOGGER SETTINGS ---
    @classmethod
    def _validate_log_wandb(cls, value, options):
        pass

    @classmethod
    def _validate_log_local(cls, value, options):
        pass

    @classmethod
    def _validate_exp_id(cls, value, options):
        pass

    @classmethod
    def _validate_run_id(cls, value, options):
        pass