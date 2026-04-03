import time
import json
import wandb
import numpy as np
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from configs.cfgparser  import Config

class Logger:
    def __init__(self, cfg: Config):
        """
        Initialize the logger and login to wandb.

        Parameters
        ----------
        cfg : Config
            Configuration object containing the following attributes:
            - `.RANK` (int): The rank of the process (0 for master).
            - `.LOG_WANDB` (bool): Whether to log to wandb.
            - `.LOG_LOCAL` (bool): Whether to log to a local file.
            - `.SAVE_MODEL` (bool): Whether to save the model.
            - `.EXP_ID` (str): The experiment ID.
            - `.RUN_ID` (str): The run ID.
            - `.LOG_JSON` (Path): The path to the local log file.
            - `.BEST_EPOCH_JSON` (Path): The path to save the best epoch log.
            - `.MODEL_PTH` (Path): The path to save the model.
            - `.TRAIN_EPOCHS` (int): The total number of training epochs.
            - `.WARMUP_EPOCHS` (int): The number of warmup epochs.
            - `.EVAL_METRIC` (str): Metric used to identify the best epoch.
        """
        self.cfg: Config = cfg
        self.rank0: bool = (cfg.RANK == 0)
        self.savedir: Path = cfg.EXP_DIR
        self.log_wandb: bool = (cfg.LOG_WANDB and self.rank0)
        self.log_local: bool = (cfg.LOG_LOCAL and self.rank0)
        self.checkpoint_interval: int = cfg.CHECKPOINT_INTERVAL
        self.exp_id: str = cfg.EXP_ID
        self.run_id: str = cfg.RUN_ID
        self.log_json: Path = cfg.LOG_JSON
        self.best_epoch_json: Path = cfg.BEST_EPOCH_JSON
        self.train_epochs: int = cfg.TRAIN_EPOCHS
        self.warmup_epochs: int = cfg.WARMUP_EPOCHS
        self.eval_metric: str = cfg.EVAL_METRIC

        self.maxacc: float = 0
        self.arrows: dict[str, str] = {
            'TTR': '\u2191', 'DSC': '\u2191', 'IoU': '\u2191',
            'ASD': '\u2193', 'HD95': '\u2193', 'AD': '\u2193', 'D95': '\u2193', 'CMA': '\u2191'
        }
        
        if self.log_wandb:
            wandb.login()
        
        if self.log_local:
            self.savedir.mkdir(parents=True, exist_ok=True)
            
    def _create_checkpoint(self, model: torch.nn.Module | DistributedDataParallel):
        """Create a checkpoint dictionary containing the model weights and config."""
        
        if isinstance(model, DistributedDataParallel):
            model = model.module

        checkpoint = {
            "weights": model.state_dict(),
            "config": self.cfg.export()
        }
        
        return checkpoint

    def init_run(self):
        """
        Initialize the wandb run and local log file.
        """
        if self.log_wandb:
            wandb.init(project = self.exp_id,
                       name = self.run_id,
                       config = self.cfg.to_dict())
        
        if self.log_local:
            self.run_summary = {}

    def end_run(self):
        """
        End the run and save the local log file.
        """
        if self.log_wandb:
            wandb.finish()

        if self.log_local:
            with self.log_json.open('w') as f:
                json.dump(self.run_summary, f)

    def set_epoch(self, epoch: int):
        """
        Set the epoch.

        Parameters
        ----------
        epoch : int
            The current epoch.
        """
        self.epoch = epoch

    def update(
        self, 
        metrics: tuple[dict, ...], 
        model: torch.nn.Module | DistributedDataParallel
    ):
        """
        Update the `epoch_log` dictionary with the values for the current epoch.
        Additionally, update the `best_epoch_log` if the main evaluation metric for the 
        current epoch outperforms the previous best. Store the model checkpoint 
        if `save_model == True`.

        Parameters
        ----------
        metrics : tuple[dict, ...]
            The training and testing metrics.
            
        model : torch.nn.Module | DistributedDataParallel
            The model to be checkpointed.
        """
        if self.rank0:
            self.trainLoss, self.testLoss, self.trainAccu, self.testAccu = metrics

            self.losses = {
                'loss/train': self.trainLoss['loss'], 
                'loss/test': self.testLoss['loss'], 
                'loss/diff': np.round(np.abs(self.trainLoss['loss'] - self.testLoss['loss']), 4)
            }
            
            self.metrics = {
                **{f'metrics-train/{k}': v for k,v in self.trainAccu.items()},
                **{f'metrics-test/{k}': v for k,v in self.testAccu.items()}
            }

            self.epoch_log = self.losses | self.metrics
            self.checkpoint = self._create_checkpoint(model)
            
            if self.testAccu[self.eval_metric] >= self.maxacc:
                self.maxacc = self.testAccu[self.eval_metric]
                self.best_epoch_log = {'epoch': self.epoch} | self.epoch_log
                
                torch.save(self.checkpoint, self.savedir / f"{self.run_id}-best.pth")
                with self.best_epoch_json.open('w') as f:
                    json.dump(self.best_epoch_log, f)
                
            if self.checkpoint_interval and self.epoch % self.checkpoint_interval == 0:
                torch.save(self.checkpoint, self.savedir / f"{self.run_id}-e{self.epoch}.pth")

    def log_metrics(self):
        """
        Log the metrics to wandb or local file.
        """
        if self.log_wandb:
            wandb.log(self.epoch_log, step = self.epoch)

        if self.log_local:
            for key, value in self.epoch_log.items():
                self.run_summary.setdefault(key, []).append(value)

    def info(self):
        """
        Print the metrics for tracking purposes.
        """
        if self.rank0:
            if self.epoch == 1 - self.warmup_epochs and self.warmup_epochs > 0:
                print(f'\n[WARM] Warming up for {self.warmup_epochs} epochs ...')

            if self.epoch <= 0:
                print(f"[WARM] Epoch {self.epoch + self.warmup_epochs}/{self.warmup_epochs}")

            if self.epoch == 1:
                print(f'\n[TRAIN] Training for {self.train_epochs} epochs ...')

            if self.epoch > 0:
                keys = self.trainAccu.keys()
                header_str = "Loss   |" + "|".join(f" {s:<{7}}" for s in [f"{self.arrows[key]} {key}" for key in keys]) + "|"
                train_metrics_str = " | ".join([f"{self.trainAccu[key]:.4f}" for key in keys])
                test_metrics_str = " | ".join([f"{self.testAccu[key]:.4f}" for key in keys])

                print(f"Epoch {self.epoch}/{self.train_epochs} > ETC: {self.etc}m ({self.dt}s / epoch)")
                print(' '*9 + header_str)
                print(f"Train -> {self.losses['loss/train']:.4f} | {train_metrics_str} |")
                print(f"Test  -> {self.losses['loss/test']:.4f} | {test_metrics_str} |")
                print('-'*(len(test_metrics_str)+19) + '+')

            if self.epoch == self.train_epochs:
                max_key_len = max(len(k) for k, _ in self.best_epoch_log.items())
                print(f'\n[EVAL] Best epoch: {self.best_epoch_log["epoch"]}')
                print("-"*(max_key_len + 9))
                for key, val in self.best_epoch_log.items():
                    if key != 'epoch':
                        if self.eval_metric in key:
                            print(f"{key.ljust(max_key_len)} : \033[30;47m{val:.4f}\033[0m")
                        else:
                            print(f"{key.ljust(max_key_len)} : {val:.4f}")
                        
                print("-"*(max_key_len + 9))

    def start_timer(self):
        """
        Start the timer for the current epoch.
        """
        if self.rank0:
            self.ts = time.time()

    def reset_timer(self):
        """
        Reset the timer for the current epoch.
        """
        if self.rank0:
            te = time.time()
            self.dt = np.round(te - self.ts)
            self.etc = np.round((self.train_epochs - self.epoch)*self.dt/60)