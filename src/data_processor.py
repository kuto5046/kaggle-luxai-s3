from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


class DataProcessor:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def read_data(self) -> None:
        self.train = pd.read_csv(Path(self.cfg.path.input_dir) / "train.csv")
        self.test = pd.read_csv(Path(self.cfg.path.input_dir) / "test.csv")

    def preprocess(self) -> None:
        pass

    def add_fold(self) -> None:
        pass

    def run(self) -> None:
        self.read_data()
        self.preprocess()
        self.add_fold()


@hydra.main(config_path="../conf", config_name="data_processor")
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
