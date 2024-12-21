from pathlib import Path

import hydra
from lightning import seed_everything
from omegaconf import DictConfig


class InferencePipeline:
    def __init__(self, cfg: DictConfig) -> None:
        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        # cfg.pathの中身をPathに変換する
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)
        self.cfg = cfg

    def setup_model(self) -> None:
        raise NotImplementedError

    def setup_dataset(self) -> None:
        raise NotImplementedError

    def inference(self) -> None:
        raise NotImplementedError

    def make_submission(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        self.setup_model()
        self.setup_dataset()
        self.inference()
        self.make_submission()


@hydra.main(config_path="../conf", config_name="inference", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = InferencePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
