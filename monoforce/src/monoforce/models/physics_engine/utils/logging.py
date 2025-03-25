from queue import Queue
from typing import Any
from dataclasses import dataclass, field
from itertools import groupby
import time
import csv
import torch
import wandb
from omegaconf import OmegaConf, DictConfig
import atexit
import threading

PROJECT = "flipper_training"


@dataclass
class RunLogger:
    train_config: DictConfig
    use_wandb: bool
    category: str
    logfiles: dict = field(default_factory=dict)
    writers: dict = field(default_factory=dict)
    log_queue: Queue = field(default_factory=Queue)
    step_metric_name: str = "log_step"
    known_wandb_metrics: set = field(default_factory=set)

    def __post_init__(self):
        ts = time.strftime("%Y-%m-%d_%H:%M:%S")
        self.logpath = ROOT / f"runs/{self.category}/{self.train_config['name']}_{ts}"
        self.logpath.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.logpath / "weights"
        self.weights_path.mkdir(exist_ok=True)
        if self.use_wandb:
            wandb.init(
                project=PROJECT,
                name=f"{self.category}_{self.train_config['name']}_{time.strftime('%Y-%m-%d_%H:%M:%S')}",
                config=OmegaConf.to_container(self.train_config, resolve=False),
                save_code=True,
            )
            wandb.define_metric(self.step_metric_name)
        self._save_config()
        atexit.register(self.close)
        self.write_thread = threading.Thread(target=self._write, daemon=True)
        self.write_thread.start()

    def _save_config(self):
        OmegaConf.save(self.train_config, self.logpath / "config.yaml")

    def _init_logfile(self, name: str, sample_row: dict[str, Any]):
        self.logfiles[name] = open(self.logpath / f"{name}.csv", "w")
        writer = csv.DictWriter(self.logfiles[name], fieldnames=[self.step_metric_name] + list(sample_row.keys()))
        self.writers[name] = writer
        writer.writeheader()
        if self.use_wandb:
            for k in sample_row.keys():
                if k not in self.known_wandb_metrics:
                    wandb.define_metric(k, step_metric=self.step_metric_name)
                    self.known_wandb_metrics.add(k)
        return writer

    def log_data(self, row: dict[str, Any], step: int):
        self.log_queue.put((step, row))

    def _write_row(self, row: dict[str, Any], step: int):
        for topic, names in groupby(row.items(), key=lambda x: x[0].rsplit("/", maxsplit=1)[0]):
            topic_row = dict(names)
            writer = self.writers.get(topic, None) or self._init_logfile(topic,topic_row)
            writer.writerow(topic_row | {self.step_metric_name: step})

    def _write(self):
        while True:
            (step, row) = self.log_queue.get()
            if self.use_wandb:
                wandb.log(data=row | {self.step_metric_name: step})
            self._write_row(row, step)

    def close(self):
        for f in self.logfiles.values():
            f.close()
        if self.use_wandb:
            wandb.finish()

    def save_weights(self, state_dict: dict, name: str):
        model_path = self.weights_path / f"{name}.pth"
        torch.save(state_dict, model_path)
        if self.use_wandb:
            wandb.log_model(
                path=model_path,
                name=name,
            )
