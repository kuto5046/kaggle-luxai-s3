import shutil
import logging
import argparse
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
    notes: str = "aDg4b, ConvLSTM"
    seed: int = 2025
    debug: bool = False
    n_splits: int = 5
    use_fold: int = 0
    root_dir: Path = Path("/kaggle")
    feature_version: str = exp_name
    feature_dir: Path = root_dir / f"output/feature_store/{feature_version}"
    output_dir = root_dir / f"exp/{exp_name}/output"

    # trainer
    epoch: int = 30
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    use_amp: bool = True
    batch_size: int = 768
    num_workers: int = 24
    ckpt_path: str = None
    lr: float = 0.001
    weight_decay: float = 0.01
    warmup_step_rate: float = 0.0  # no warmup
    # model
    num_repeats = 1
    num_layers = 4
    hidden_dim = 64
    n_stack: int = 4
    
    # loss
    loss_weight_policy: float = 1.0
    loss_weight_state: float = 1.0
    loss_weight_global_state: float = 0.0
    # loss_weight_value: float = 0.0
    # loss_weight_sap: float = 1.0

    @classmethod
    def from_args(cls) -> "Config":
        """
        configをコマンドライン引数から読み込む
        $ uv run python exp/exp016/train.py --debug --epoch 2
        """
        parser = argparse.ArgumentParser()
        for field in cls.__dataclass_fields__.values():
            parser.add_argument(
                f"--{field.name}",
                type=field.type,
                default=field.default,
                help=f"{field.name} (default: {field.default})",
            )
        args = parser.parse_args()
        return cls(**vars(args))


class TrainPipeline:
    def __init__(self, cfg: Config) -> None:
        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.output_dir = cfg.output_dir
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.cfg = cfg
        self.debug_config()

    def debug_config(self) -> None:
        if self.cfg.debug:
            self.cfg.epoch = 2
            self.cfg.limit_train_batches = 0.01
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
        if not self.cfg.debug:
            self.pl_logger = WandbLogger(
                project="kaggle-luxai-s3",
                entity="okumura",
                # name=f"{self.cfg.exp_name}",
                group=self.cfg.exp_name,
                mode="disabled" if self.cfg.debug else "online",
                notes=self.cfg.notes,
            )
        else:
            self.pl_logger = None

    def setup_model(self) -> None:
        if self.cfg.ckpt_path:
            self.model = LaxLitModel.load_from_checkpoint(self.cfg.ckpt_path, cfg=self.cfg)
        else:
            self.model = LaxLitModel(self.cfg)
        
    def train(self) -> None:
        self.trainer = Trainer(
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
        self.trainer.fit(self.model, datamodule=self.datamodule)

    def run(self) -> None:
        self.setup_logger()
        self.setup_dataset()
        self.setup_callbacks()
        self.setup_model()
        self.train()
        self.pl_logger.finalize(status="success")


def main() -> None:
    cfg = Config.from_args()
    pipeline = TrainPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
