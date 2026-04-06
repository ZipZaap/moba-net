from typing import Any
from pathlib import Path
from configs.validator import Validator

class Config:
    """
    A configuration loader that reads key:value pairs from a parameter dictionary
    and sets them as attributes. After initialization, attributes are
    immutable unless their names are listed in `MUTABLE_KEYS`.
    """

    EXPORT_KEYS: list[str] = ["MODEL", "UNET_DEPTH", "CONV_DEPTH", "INPUT_CHANNELS", "SEG_CLASSES", "CLS_CLASSES"]
    MUTABLE_KEYS: list[str] = ["RANK", "DEVICE"]

    def __init__(self, 
                 cfg: dict, 
                 *, 
                 inference: bool = False
                 ):
        """
        Initialize the configuration object.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary containing key:value pairs.

        inference : bool
            Whether the configuration is for inference (default: False).

        Examples
        --------
        >>> import yaml
        >>> from pathlib import Path
        >>>
        >>> with Path("configs/config.yaml").open() as f:
        >>>     cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
        >>>
        >>> CONF = Config(cfg, inference=True)
        """

        # Immutability disabled by default
        self._frozen: bool = False
        self._inference: bool = inference

        # Validate the configuration
        Validator.validate_cfg(cfg, inference)

        # Set attributes from <cfg> dict
        for key, value in cfg.items():
            setattr(self, key, value['default'])

        # Set dependent attributes
        self._set_dependent_attributes()

        # Mark as initialized; freeze attributes
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value):
        """
        Set an attribute on the Config object.
        If the object is frozen, only mutable keys can be changed.
        """

        if name.startswith('_') or name in self.MUTABLE_KEYS or not getattr(self, '_frozen', False):
            return object.__setattr__(self, name, value)
        raise AttributeError(f"Cannot modify attribute '{name}'; it's immutable.")
    
    def __getattr__(self, name: str) -> Any:
        """
        Called when an attribute lookup fails in the normal places
        (i.e. it's not found in __dict__ or via __getattribute__).
        We check our __dict__ and return it to satisfy Pylance.
        """

        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self):
        """
        String representation of the Config object, showing its attributes.
        """

        attrs = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        return f"<Config {attrs}>"
    
    def _exp_id(self) -> str:
        """
        Generate an experiment ID based on the highest existing exp_N in the results directory.

        Returns
        -------
        id : str
            Next available experiment ID (e.g., 'exp_7' if 'exp_6' exists).
        """

        exp_nums = []
        if self.RESULTS_DIR.exists():
            for d in self.RESULTS_DIR.iterdir():
                if d.is_dir() and d.name.startswith("exp_"):
                    num = int(d.name.split("_")[1])
                    exp_nums.append(num)
        id = max(exp_nums, default=-1) + 1

        return f'exp_{id}'

    def _set_dependent_attributes(self):
        """
        Set dependent attributes based on the loaded configuration.
        This includes paths, derived numerical values, and model-specific settings.
        """

        # Core paths -------------------------------------------------------------------
        self.DATASET_DIR = Path(self.DATASET_DIR)
        self.CHECKPOINT = Path(self.CHECKPOINT) if self.CHECKPOINT else None

        # Derived values ---------------------------------------------------------------
        self.DEFAULT_DEVICE = f'cuda:{self.GPUs[0]}' if self.GPUs else 'cuda:0'

        if self._inference:
            # Prediction dataset paths -------------------------------------------------
            self.PREDICT_DIR = self.DATASET_DIR / 'predict'
            self.IMG_DIR = self.PREDICT_DIR / 'images'
            self.MSK_DIR = self.PREDICT_DIR / 'masks'

        else:
            # Basic paths --------------------------------------------------------------
            self.BASE_IMG_DIR = self.DATASET_DIR / 'images'
            self.BASE_MSK_DIR = self.DATASET_DIR / 'masks'
            self.LBL_JSON = self.DATASET_DIR / 'labels.json'

            # Training dataset paths ---------------------------------------------------
            self.TRAIN_DIR = self.DATASET_DIR / 'train'
            self.IMG_DIR = self.TRAIN_DIR / 'images'
            self.MSK_DIR = self.TRAIN_DIR / 'masks'
            self.SDM_DIR = self.TRAIN_DIR / 'sdms'
            self.TTS_JSON = self.TRAIN_DIR / 'tts.json'

            # Results paths & ID strings -----------------------------------------------
            self.RESULTS_DIR = Path(self.RESULTS_DIR)
            self.EXP_ID = self.EXP_ID if self.EXP_ID else self._exp_id()
            self.RUN_ID = self.RUN_ID if self.RUN_ID else f"{self.MODEL}({self.TRAIN_SET})"
            self.EXP_DIR = self.RESULTS_DIR / self.EXP_ID
            self.LOG_JSON = self.EXP_DIR / f"{self.RUN_ID}-log.json"
            self.BEST_EPOCH_JSON = self.EXP_DIR / f"{self.RUN_ID}-best.json"
    
            # Derived values -----------------------------------------------------------
            self.NUM_KFOLDS = int(1 / self.TEST_SPLIT)
            self.WORLD_SIZE = len(self.GPUs)
            self.BATCH_SIZE = int(self.BATCH_SIZE / self.WORLD_SIZE)

            # Model-specific settings --------------------------------------------------
            model_freeze_layers = {
                'MobaNet_EDC': [],
                'MobaNet_ED': ['classifier'],
                'MobaNet_EC': ['decoder'],
                'MobaNet_C': ['encoder', 'decoder'],
                'MobaNet_D': ['encoder', 'classifier'],
                'UNet': [],
            }
            self.FREEZE_LAYERS = model_freeze_layers[self.MODEL]
    
    def export(self) -> dict[str, Any]:
        """
        Export the model-specific parameter values. 
        Later needed to load the model for inference.

        Returns
        -------
        model_cfg : dict[str, Any]
            Dictionary containing the model-specific parameters.
        """

        model_cfg: dict[str, Any] = {}
        for k in Config.EXPORT_KEYS:
            model_cfg[k] = getattr(self, k)

        return model_cfg
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the configuration to a dictionary representation.

        Returns
        -------
        cfg_dict : dict[str, Any]
            Dictionary containing all the public `Config` attributes.
        """
        
        cfg_dict = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if isinstance(v, Path):
                    cfg_dict[k] = str(v)
                else:
                    cfg_dict[k] = v
        return cfg_dict