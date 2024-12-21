import logging
from typing import Any
from pathlib import Path

import hydra
import wandb
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


class TrainPipeline:
    def __init__(self, cfg: DictConfig) -> None:
        # cfg.pathの中身をPathに変換する
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする

        # hydraのrun_dirに同じpathが設定されているので自動でディレクトリが作成される
        self.output_dir = cfg.path.output_dir / cfg.exp_name / cfg.run_name

        self.cfg = cfg
        self.models: list[Any] = []
        self.oofs: list[pl.DataFrame] = []
        self.scores: list[float] = []
        assert cfg.phase == "train", "TrainPipeline only supports train phase"

    def setup_dataset(self, fold: int) -> None:
        pass

    def setup_callbacks(self) -> list:
        return []

    def setup_logger(self, fold: int) -> None:
        wandb.init(
            project="kaggle-template",
            entity="kuto5046",
            name=f"{self.cfg.exp_name}_{self.cfg.run_name}_fold{fold}",
            group=self.cfg.exp_name,
            tags=self.cfg.tags,
            mode="disabled" if self.cfg.debug else "online",
            notes=self.cfg.notes,
        )

    def train(self, fold: int) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def run(self) -> None:
        for fold in self.cfg.use_folds:
            self.setup_logger(fold)
            self.setup_dataset(fold)
            self.train(fold)
            self.evaluate()
            # 最後のfold以外であればwandbをfinishする
            if fold != self.cfg.use_folds[-1]:
                wandb.finish()

        oof = pl.concat(self.oofs).sort("Id")
        oof.write_csv(self.output_dir / "oof.csv")
        wandb.finish()


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = TrainPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
