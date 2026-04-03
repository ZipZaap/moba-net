import os
import yaml
import inspect
import itertools
from pathlib import Path
from typing import Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.optim import Adam
from torch.multiprocessing.spawn import spawn
from torch.optim.lr_scheduler import LinearLR

from utils.sdf import SDF
from utils.loggers import Logger
from utils.dataset import DatasetTools
from utils.managers import ProcessManager
from engines.SegTrainer import SegTrainer

from model.metrics import Accuracy
from model.loss import Loss

from configs.cfgparser import Config
from configs.cli import parse_cli_args

def _main_worker(rank: int, cfg: Config):
    """
    Main function to run the training process.

    Parameters
    ----------
    rank : int
        The rank of the process.

    cfg : Config
        Configuration object containing the training parameters.
    """
    
    # Bind rank to config and device
    cfg.RANK = rank
    process = ProcessManager(cfg)
    process.bind_to_device()

    model = process.load_model()
    loaders = DatasetTools.train_dataloaders(cfg)
    logger = Logger(cfg)
    loss = Loss(cfg)
    accu = Accuracy(cfg)
        
    optimizer = Adam([
        # Group 1: main model parameters
        {
            "params": filter(lambda p: p.requires_grad, model.parameters()),
            "weight_decay": cfg.L2_DECAY,
            "lr": cfg.BASE_LR,
        },
        # Group 2: adaptive loss weights
        {
            "params": loss.parameters(),
            'weight_decay': 1e-3,
            "lr": cfg.BASE_LR,
        }
    ])

    lr_scheduler = LinearLR(
        optimizer, 
        start_factor=cfg.INIT_LR/cfg.BASE_LR, 
        total_iters=cfg.WARMUP_EPOCHS
    )

    SegTrainer(model, optimizer, lr_scheduler, 
               loaders, logger, loss, accu, cfg).train()
    
    process.cleanup()


def train(
    cfg: dict | str,
    *,
    dataset_dir: Optional[str] = None,
    train_set: Optional[str] = None,
    test_set: Optional[str] = None,
    model: Optional[str] = None,
    checkpoint: Optional[str] = None,
    batch_size: Optional[int] = None,
    train_epochs: Optional[int] = None,
    loss: Optional[str] = None,
    GPUs: Optional[list[int]] = None
):
    """
    Launch training programmatically (importable API), or from a notebook/script. 
    The function exposes only the basic configuration options. For a more granular control, 
    consider modifying the `config.yaml` file directly or using the CLI.

    Parameters
    ----------
    config : dict | str
        Either a loaded parameter dictionary or a path to a `config.yaml` file.

    dataset_dir : str, optional
        Path to the dataset directory.

    train_set : str, optional
        Composition of the training set.

    test_set : str, optional
        Composition of the testing set.

    model : str, optional
        Name of the model to use.

    checkpoint : str, optional
        Path to the model checkpoint.

    batch_size : int, optional
        Batch size for training.

    train_epochs : int, optional
        Number of training epochs.

    loss : str, optional
        Loss function to use.

    GPUs : list[int], optional
        List of GPU IDs to use for training

    Raises
    ------
    RuntimeError
        If no GPU with CUDA support is available.

    ValueError
        If cfg is not a valid Config object or YAML file.

    Examples
    --------
    Using a config file path directly:

    >>> from train import train
    >>> train("configs/config.yaml", dataset_dir="path/to/dataset",
    ...       batch_size=16, train_epochs=100, loss="SoftDICE", GPUs=[0, 1])

    Loading the config with PyYAML first & modifying the parameters (e.g from CLI):

    >>> import yaml
    >>> from train import train
    >>> with open("configs/config.yaml") as f:
    ...     cfg = yaml.load(f, Loader=yaml.FullLoader)
    >>> cfg = parse_cli_args(cfg, inference=False)
    >>> train(cfg)
    """

    # Basic environment check
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This library requires a GPU with CUDA support. "
            "Please verify the PyTorch installation and ensure that a compatible GPU is available."
        )
    
    # Load cfg dictionary 
    if isinstance(cfg, dict):
        _cfg: dict = cfg
    elif isinstance(cfg, str) and os.path.exists(cfg) and cfg.endswith('.yaml'):
        with Path(cfg).open() as f:
            _cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError("cfg should either be a parameter dictionary or a path to a valid YAML file.")

    # Isolate arguments provided through the function signature
    sig = inspect.signature(train)
    params = set(sig.parameters.keys())
    local_args = {
        k: v for k, v in locals().items()
        if k in params and k != "cfg" and v is not None
    }

    # Update config with local variables
    for param, value in local_args.items():
        _cfg[param.upper()]['default'] = value

    # Instantiate config object
    CONF: Config = Config(_cfg)

    # Setup
    DatasetTools.compose_dataset(CONF)
    SDF.generate_sdms(CONF)

    # Launch
    if CONF.WORLD_SIZE > 1:
        print(f"[INFO] Running in distributed mode on {CONF.WORLD_SIZE} GPUs ...")
        os.environ['NCCL_P2P_DISABLE'] = '0' if CONF.NCCL_P2P else '1'
        spawn(_main_worker, args=(CONF,), nprocs=CONF.WORLD_SIZE)
    else:
        print(f"[INFO] Running in non-distributed mode on {CONF.DEFAULT_DEVICE} ...")
        _main_worker(0, CONF)

if __name__ == "__main__":
    with Path("configs/config.yaml").open() as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)

    cfg = parse_cli_args(cfg, inference=False)

    train(cfg)