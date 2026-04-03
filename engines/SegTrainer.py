import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.loss import Loss
from utils.loggers import Logger
from model.metrics import Accuracy
from configs.cfgparser  import Config

class SegTrainer():
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer: Optimizer, 
                 lr_scheduler: LRScheduler, 
                 loaders: tuple[DataLoader, DataLoader],
                 logger: Logger,
                 loss: Loss,
                 accuracy: Accuracy,
                 cfg: Config):
        """
        Initializes the SegTrainer. This class is responsible for training and evaluating the model.
        It handles the training loop, validation, and logging of metrics.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be trained.

        optimizer : Optimizer
            Optimizer for the model.

        lr_scheduler : LRScheduler
            Learning rate scheduler.

        loaders : tuple[DataLoader, DataLoader]
            Tuple of train and test loaders.

        logger : Logger
            Logger for logging metrics and saving model checkpoints.
        
        loss : Loss
            Loss function for training.
        
        accuracy : Accuracy
            Accuracy metric for evaluation.

        cfg : Config
            Configuration object that contains the following attributes:
            - `.DEVICE` (str): Device to run the model on.
            - `.TRAIN_EPOCHS` (int): Number of training epochs.
            - `.WARMUP_EPOCHS` (int): Number of warmup epochs.
        """

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.trainLoader, self.testLoader = loaders
        self.logger = logger
        self.loss = loss
        self.accu = accuracy

        self.device: str = cfg.DEVICE
        self.train_epochs: int = cfg.TRAIN_EPOCHS
        self.warmup_epochs: int = cfg.WARMUP_EPOCHS
        
    def _set_model_state(self, train: bool):
        """
        Safely sets the model to train or eval mode, ensuring frozen 
        sub-modules remain in eval mode to protect BatchNorm/Dropout.
        
        Parameters
        ----------
        train : bool
            If True, sets model to train mode; if False, sets to eval mode.
        """
        if train:
            self.model.train() 
            for child in self.model.children():
                params = list(child.parameters())
                if params and not any(p.requires_grad for p in params):
                    child.eval() 
        else:
            self.model.eval()
            
    def _warmup_epoch(self, loader: DataLoader):
        """
        Warmup training for the model by gradually increasing the learning rate.

        Parameters
        ----------
        loader : DataLoader
            DataLoader for the training data.
        """
        
        self._set_model_state(train=True)
        
        with torch.set_grad_enabled(True):

            for batch in loader:
                
                batch = {k:v.to(self.device) for k,v in batch.items()}
                logits = self.model(batch['image'])

                self.optimizer.zero_grad()
                self.loss.update(logits, batch)
                self.loss.backprop()
                self.optimizer.step()

        self.lr_scheduler.step()

    def _learn_epoch(self,
                     loader: DataLoader,
                     train: bool,
                     ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Runs a single epoch of training or evaluation

        Parameters
        ----------
        loader : DataLoader
            DataLoader for the training or evaluation data.

        train : bool
            If True, runs training; if False, runs evaluation.
            
        Returns
        -------
        avgLoss : dict[str, float]
            Average loss for the epoch.
            
        avgAccu : dict[str, float]
            Average accuracy for the epoch.
        """
        
        self._set_model_state(train)

        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = {k:v.to(self.device) for k,v in batch.items()}
                logits = self.model(batch['image'])

                self.accu.update(logits, batch) 
                self.loss.update(logits, batch)

                if train:
                    self.optimizer.zero_grad()
                    self.loss.backprop()
                    self.optimizer.step()

        avgLoss = self.loss.compute_avg(loader.__len__())
        self.loss.reset()

        avgAccu = self.accu.compute_avg(loader.__len__())
        self.accu.reset()
        
        return avgLoss, avgAccu

    def train(self):
        """
        Main training loop; iterates over epochs and runs warmup, training & evaluation.
        """
        
        self.logger.init_run()
        for epoch in range(-self.warmup_epochs + 1, self.train_epochs + 1):
            
            self.logger.set_epoch(epoch)
            if isinstance(self.trainLoader.sampler, DistributedSampler):
                self.trainLoader.sampler.set_epoch(epoch)
            
            if epoch <= 0:
                self._warmup_epoch(self.trainLoader)
            else:
                self.logger.start_timer()
                trainLoss, trainAccu = self._learn_epoch(self.trainLoader, train=True)
                testLoss, testAccu = self._learn_epoch(self.testLoader, train=False)
                self.logger.update((trainLoss, testLoss, trainAccu, testAccu), self.model)    
                self.logger.log_metrics()
                self.logger.reset_timer()

            self.logger.info()

        self.logger.end_run()