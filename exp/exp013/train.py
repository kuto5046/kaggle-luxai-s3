import shutil
import logging
from pathlib import Path
from dataclasses import dataclass

import seaborn as sns
from lightning import Trainer, seed_everything
from lux.models import LaxLitModel, LaxLitDataModule
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers.wandb import WandbLogger

sns.set_theme()

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    exp_name: str = Path(__file__).parent.name
    notes: str = "hidden stateを補助lossに使う all data"
    seed: int = 2025
    debug: bool = False
    n_splits: int = 5
    use_fold: int = 0
    root_dir: Path = Path("/home/user/work")
    feature_version: str = "exp012"
    feature_dir: Path = root_dir / f"output/feature_store/{feature_version}"
    output_dir = root_dir / f"exp/{exp_name}/output"

    epoch: int = 10
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    use_amp: bool = False
    batch_size: int = 512
    num_workers: int = 12
    ckpt_path: str = None
    lr: float = 0.001
    weight_decay: float = 0.01
    warmup_step_rate: float = 0.1

    n_stack: int = 4


class TrainPipeline:
    def __init__(self, cfg: Config) -> None:
        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        shutil.rmtree(self.output_dir)

        self.cfg = cfg
        self.debug_config()

    def debug_config(self) -> None:
        if self.cfg.debug:
            self.cfg.epoch = 2
            self.cfg.limit_train_batches = 0.1
            self.cfg.limit_val_batches = 0.1

    def setup_dataset(self) -> None:
        self.datamodule = LaxLitDataModule(self.cfg)

    def setup_callbacks(self) -> None:
        epoch_checkpoint = ModelCheckpoint(
            dirpath=self.output_dir,
            monitor="Loss/valid",
            mode="min",
            filename="best_model",
            save_weights_only=True,
            verbose=True,
        )
        lr_monitor = LearningRateMonitor("step")
        progress_bar = RichProgressBar()
        model_summary = RichModelSummary(max_depth=2)
        self.callbacks = [
            epoch_checkpoint,
            lr_monitor,
            progress_bar,
            model_summary,
        ]

    def setup_logger(self) -> None:
        self.pl_logger = WandbLogger(
            project="kaggle-luxai-s3",
            entity="kuto5046",
            # name=f"{self.cfg.exp_name}",
            group=self.cfg.exp_name,
            mode="disabled" if self.cfg.debug else "online",
            notes=self.cfg.notes,
        )

    def setup_model(self) -> None:
        self.model = LaxLitModel(self.cfg)

    def train(self) -> None:
        self.trainer = Trainer(
            # env
            # default_root_dir=Path.cwd(),
            accelerator="auto",
            precision="16-mixed" if self.cfg.use_amp else 32,
            max_epochs=self.cfg.epoch,
            callbacks=self.callbacks,
            logger=self.pl_logger,
            num_sanity_val_steps=0,
            sync_batchnorm=True,
            limit_train_batches=self.cfg.limit_train_batches,
            limit_val_batches=self.cfg.limit_val_batches,
            deterministic=True,  # for reproducibility
        )
        self.trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=self.cfg.ckpt_path)

    def run(self) -> None:
        self.setup_logger()
        self.setup_dataset()
        self.setup_callbacks()
        self.setup_model()
        self.train()
        self.pl_logger.finalize(status="success")


def main() -> None:
    cfg = Config()
    pipeline = TrainPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
