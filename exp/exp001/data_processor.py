import json
import shutil
import logging
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass

import h5py
import numpy as np
import polars as pl
from lightning import seed_everything
from lux.utils import extract_state, extract_action
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    exp_name: str = Path(__file__).parent.name
    seed: int = 2025
    debug: bool = False
    phase: str = "train"  # train, test
    n_splits: int = 5
    root_dir: Path = Path("/home/user/work")
    input_dir: Path = root_dir / "input"
    episode_dir: Path = input_dir / "kuto-luxai-s3-episodes-20250104"
    feature_dir: Path = root_dir / f"output/feature_store/{exp_name}"
    target_team_name: str = "ry_andy_"
    target_sub_ids: list[int] = field(default_factory=lambda: [42165330])


def get_fold(_train: pl.DataFrame, cv: list[tuple[np.ndarray, np.ndarray]]) -> pl.DataFrame:
    """
    trainにfoldのcolumnを付与する
    """
    train = _train.clone()
    train = train.with_columns(pl.lit(-1).alias("fold"))
    for fold, (train_idx, valid_idx) in enumerate(cv):
        train = train.with_columns(
            pl.when(pl.arange(0, len(train)).is_in(valid_idx)).then(fold).otherwise(pl.col("fold")).alias("fold")
        )
    LOGGER.info(train.group_by("fold").len().sort("fold"))
    return train


def get_kfold(train: pl.DataFrame, n_splits: int, seed: int = 0) -> pl.DataFrame:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = list(kf.split(X=train))
    return get_fold(train, cv)


def valid_episode(json_load: dict[str, Any], target_team_name: str) -> bool:
    """対象のチームが勝利してるepisodeのみ有効"""
    win_idx = np.argmax([r or 0 for r in json_load["rewards"]])  # win or tie
    win_team = json_load["info"]["TeamNames"][win_idx]
    return win_team == target_team_name


class DataProcessor:
    def __init__(self, cfg: Config) -> None:
        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.cfg = cfg
        self.episode_dir = cfg.episode_dir
        self.feature_dir = cfg.feature_dir
        if self.feature_dir.exists():
            shutil.rmtree(self.feature_dir)
            print(f"remove {self.feature_dir}")
        self.feature_dir.mkdir(parents=True, exist_ok=True)

    def read_data(self) -> pl.DataFrame:
        episode_df = pl.read_csv(self.episode_dir / "episodes.csv")
        episode_df = episode_df.filter(pl.col("SubmissionId").is_in(self.cfg.target_sub_ids))
        print(f"episode_df: {len(episode_df)}")
        # なぜかepisodeに重複があるため除去
        episode_df = episode_df.unique("EpisodeId")
        print(f"unique episode_df: {len(episode_df)}")
        if self.cfg.debug:
            episode_df = episode_df.sample(n=10)
        return episode_df

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        valid_ids = []
        max_steps = []
        with h5py.File(self.feature_dir / "episodes.h5", "w") as out_f:
            for row in tqdm(df.iter_rows(named=True), total=len(df)):
                sub_id = row["SubmissionId"]
                episode_id = row["EpisodeId"]
                episode_path = self.episode_dir / f"{sub_id}/{episode_id}.json"

                with open(episode_path) as f:
                    json_load = json.load(f)

                # 無効なepisodeはスキップ
                if not valid_episode(json_load, self.cfg.target_team_name):
                    continue
                valid_ids.append(str(episode_id))

                episode_group = out_f.create_group(f"{episode_id}")
                episode_action_group = episode_group.create_group("actions")
                episode_state_group = episode_group.create_group("states")
                target_team_idx = np.argmax([r or 0 for r in json_load["rewards"]])  # win or tie
                steps = json_load["steps"]
                for step_idx, step_info in enumerate(steps):
                    # DONEのstep以降は無視
                    if step_info[target_team_idx]["status"] == "DONE":
                        max_steps.append(str(step_idx))
                        break
                    state = extract_state(step_info, target_team_idx)
                    action = extract_action(step_info, target_team_idx)
                    episode_action_group.create_dataset(f"{step_idx}", data=action)
                    episode_state_group.create_dataset(f"{step_idx}", data=state)
        return pl.DataFrame(
            {"EpisodeId": valid_ids, "MaxStep": max_steps},
            # h5pyのkeyとして使うためにstrに変換
            schema={"EpisodeId": pl.String, "MaxStep": pl.String},
        )

    def add_fold(self, df: pl.DataFrame) -> pl.DataFrame:
        return get_kfold(df, self.cfg.n_splits, self.cfg.seed)

    def run(self) -> None:
        episode_paths = self.read_data()
        df = self.preprocess(episode_paths)
        df = self.add_fold(df)
        df.write_csv(self.feature_dir / "train.csv")


def main() -> None:
    cfg = Config()
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
