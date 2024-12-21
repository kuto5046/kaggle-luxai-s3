## 再現性のテスト
import os
from pathlib import Path

from hydra import compose, initialize
from omegaconf import DictConfig

from src.train import TrainPipeline


def setup_train_config() -> DictConfig:
    # 強制的にdebugモードにする
    with initialize(version_base="1.2", config_path="../conf", job_name="test_pipeline"):
        cfg = compose(
            config_name="train",
            overrides=["debug=True", "run_name=test_pipeline"],
            return_hydra_config=True,
        )
        # hydraのchdirを手動で行う必要がある
        if cfg.hydra.get("run", {}).get("dir", False):
            run_dir = Path(cfg.hydra.run.dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(run_dir)
            print(f"Current working directory is changed to {os.getcwd()}")
    return cfg


def get_trainpipeline_result(cfg: DictConfig) -> float | None:
    pipeline = TrainPipeline(cfg)
    pipeline.run()
    return pipeline.cv_score


# def get_dataset_result(cfg: DictConfig) -> LightningDataModule:
#     pipeline = TrainPipeline(cfg)
#     pipeline.setup_debug_config()
#     pipeline.setup_logger()
#     pipeline.setup_dataset()
#     pipeline.datamodule.setup()
#     return pipeline.datamodule


def test_dataset_reproduce() -> None:
    pass
    # cfg = setup_train_config()
    # datamodule1 = get_dataset_result(cfg)
    # datamodule2 = get_dataset_result(cfg)

    # データが同じかどうかを確認する
    # assert_frame_equal(datamodule1.train, datamodule2.train)
    # assert_frame_equal(datamodule1.valid, datamodule2.valid)


def test_trainpipeline_reproduce() -> None:
    """
    train pipelineをe2eで実行して、cv scoreが再現性があるかを確認する
    """
    cfg = setup_train_config()
    score1 = get_trainpipeline_result(cfg)
    score2 = get_trainpipeline_result(cfg)
    assert score1 == score2, f"not reproduce cv score: {score1} != {score2}"
