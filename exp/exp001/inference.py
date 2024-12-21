import logging
from pathlib import Path

import hydra
from lightning import seed_everything
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self, cfg: DictConfig) -> None:
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.cfg = cfg
        self.model_dir = self.cfg.path.model_dir / self.cfg.exp_name / self.cfg.run_name
        assert cfg.phase == "test", "InferencePipeline only supports test phase"

    def read_data(self) -> None:
        pass

    def setup_model(self) -> None:
        pass

    def setup_dataset(self) -> None:
        pass

    def inference(self) -> None:
        pass

    def make_submission(self) -> None:
        pass

    def run(self) -> None:
        self.read_data()
        # preds = []
        for fold in self.cfg.use_folds:
            self.setup_model()
            self.setup_dataset()
            self.inference()
            # preds.append(pred)
        self.make_submission()


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = InferencePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
