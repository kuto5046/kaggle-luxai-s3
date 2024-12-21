import logging
from pathlib import Path

import hydra
from lightning import seed_everything
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


class TrainPipeline:
    def __init__(self, cfg: DictConfig) -> None:
        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        # cfg.pathの中身をPathに変換する
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        self.cfg = cfg
        self.cv_score: float | None = None

    def setup_debug_config(self) -> None:
        # debug時はwandbを無効にするようにsetup_loggerで行っている
        if self.cfg.debug:
            pass

    def setup_dataset(self) -> None:
        raise NotImplementedError

    def setup_callbacks(self) -> None:
        raise NotImplementedError

    def setup_logger(self) -> None:
        raise NotImplementedError

    def setup_model(self) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        self.setup_debug_config()
        self.setup_logger()
        self.setup_dataset()
        self.setup_callbacks()
        self.setup_model()
        self.train()
        self.evaluate()
        # self.pl_logger.finalize(status="success" if self.cv_score is not None else "failed")


@hydra.main(config_path="../conf", config_name="train", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = TrainPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
