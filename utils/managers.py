import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.MobaNet import MobaNet
from configs.cfgparser  import Config

class ProcessManager():
    def __init__(self, cfg: Config):
        """
        Initialize the subprocess for distributed training.

        Parameters
        ----------
        cfg : Config
            Configuration object containing the following attributes:
            1. General
            - `.RANK` (int): The rank of the process.
            - `.EXP_DIR` (Path): Save directory of the current experiment.
            - `._inference` (bool): Whether the model is in inference mode.

            2. DDP
            - `.GPU_LIST` (list[int]): List of GPUs to use.
            - `.WORLD_SIZE` (int): Number of processes (GPUs) in the distributed training.
            - `.MASTER_ADDR` (str): Master address for distributed training.
            - `.MASTER_PORT` (str): Master port for distributed training.

            3. Model-specific
            - `.MODEL` (str): Name of the model architecture.
            - `.UNET_DEPTH` (int): Depth of the U-Net architecture.
            - `.CONV_DEPTH` (int): Depth of the convolutional layers.
            - `.INPUT_CHANNELS` (int): Number of input channels.
            - `.SEG_CLASSES` (int): Number of segmentation classes.
            - `.CLS_CLASSES` (int): Number of classification classes.
            - `.SEG_DROPOUT` (float | None): Dropout rate for segmentation layers.
            - `.CLS_DROPOUT` (float | None): Dropout rate for classification layers.
            - `.CHECKPOINT` (str | None): Name of the checkpoint file (without .pth extension).
            - `.FREEZE_LAYERS` (list[str]): List of layers to freeze.
        """
        
        # general attributes
        self.cfg: Config = cfg
        self.rank: int = cfg.RANK
        self.exp_dir: Path = cfg.EXP_DIR

        # distributed training attributes
        self.gpu_list: list[int] = cfg.GPUs
        self.worldsize: int = cfg.WORLD_SIZE
        self.master_addr: str = cfg.MASTER_ADDR
        self.master_port: str = cfg.MASTER_PORT

        # model-architecture attributes
        self.model : str = cfg.MODEL
        self.unet_depth: int = cfg.UNET_DEPTH
        self.conv_depth: int = cfg.CONV_DEPTH
        self.in_channels: int = cfg.INPUT_CHANNELS
        self.seg_classes: int = cfg.SEG_CLASSES
        self.cls_classes: int = cfg.CLS_CLASSES
        self.seg_dropout: float = cfg.SEG_DROPOUT
        self.cls_dropout: float = cfg.CLS_DROPOUT

        # pre-trained weights loading attributes
        self.checkpoint: Path | None = cfg.CHECKPOINT
        self.freeze_layers: list[str] = cfg.FREEZE_LAYERS

    def bind_to_device(self):
        """
        Set the device and initialize distributed if needed.
        """
     
        if self.worldsize > 1:
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = self.master_port

            self.gpu_id = self.gpu_list[self.rank]
            self.device = f"cuda:{self.gpu_id}"
            torch.cuda.set_device(self.gpu_id)

            init_process_group(
                backend="nccl",
                rank=self.rank,
                world_size=len(self.gpu_list)
            )
        else:
            self.gpu_id = self.gpu_list[0]
            self.device = f"cuda:{self.gpu_id}"
            torch.cuda.set_device(self.gpu_id)

        self.cfg.DEVICE = self.device

    def cleanup(self):
        """
        Cleanup the process group for distributed training.
        """
        if self.worldsize > 1:
            destroy_process_group()

    def load_model(self):
        """
        Load the model and its pre-trained weights (if specified).

        Returns
        -------
            model : torch.nn.Module
                The loaded model.
        """
        
        model = MobaNet(
            model = self.model,
            unet_depth = self.unet_depth,
            conv_depth = self.conv_depth,
            in_channels = self.in_channels,
            seg_classes = self.seg_classes,
            cls_classes = self.cls_classes,
            seg_dropout = self.seg_dropout,
            cls_dropout = self.cls_dropout
        )

        if self.checkpoint:
            if self.rank == 0:
                print(f"[INFO] Loading checkpoint: {self.checkpoint.stem}")
                
            weights = torch.load(self.checkpoint, 
                                 map_location=self.device, 
                                 mmap=True, 
                                 weights_only=True)['weights']
            model.load_state_dict(weights, strict=False)     
        
        if self.freeze_layers and self.rank == 0:
            print(f"[INFO] Freezing layers: {self.freeze_layers}")
                
        for layer_name in self.freeze_layers:
            layer = getattr(model, layer_name)
            for param in layer.parameters():
                param.requires_grad = False
            

        if self.worldsize > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model.to(self.device), 
                        device_ids=[self.gpu_id], 
                        find_unused_parameters=False)
        else:
            model = model.to(self.device)

        return model