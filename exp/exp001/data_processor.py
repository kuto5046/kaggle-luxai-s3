from pathlib import Path

import hydra
from lightning import seed_everything
from omegaconf import DictConfig


class DataProcessor:
    def __init__(self, cfg: DictConfig) -> None:
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.cfg = cfg

    def read_data(self) -> None:
        pass

    def preprocess(self) -> None:
        pass

    def add_fold(self) -> None:
        pass

    def run(self) -> None:
        self.read_data()
        self.preprocess()
        self.add_fold()


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
