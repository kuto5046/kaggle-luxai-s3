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
from lux.utils import EpisodeStore, extract_state, extract_action
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    exp_name: str = Path(__file__).parent.name
    seed: int = 2025
    debug: bool = False
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
            episode_df = episode_df.sample(n=10, seed=self.cfg.seed)
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

                # episode_hidden_state_group = episode_group.create_group("hidden_states")
                target_team_id = np.argmax([r or 0 for r in json_load["rewards"]])  # win or tie

                # episode内で獲得する情報
                episode_store = EpisodeStore(target_team_id)
                episode_store.load_env_cfg(json_load["configuration"]["env_cfg"])
                steps = json_load["steps"]
                for step_idx in range(len(steps) - 1):
                    prev_step_info = steps[step_idx - 1] if step_idx > 0 else None
                    step_info = steps[step_idx]
                    next_step_info = steps[step_idx + 1]
                    obs = json.loads(step_info[target_team_id]["observation"]["obs"])

                    # マッチごとにリセットされる要素をリセット
                    if obs["match_steps"] == 0:
                        episode_store.reset()

                    # prev_actions = step_info[target_team_id]["action"]
                    if prev_step_info is not None:
                        prev_actions = prev_step_info[target_team_id]["action"]
                    else:
                        prev_actions = {}
                    episode_store.update(obs, prev_actions)

                    state = extract_state(obs, target_team_id, episode_store)
                    episode_state_group.create_dataset(f"{step_idx}", data=state)

                    # gt_obs = step_info[0]["info"]["replay"]["observations"][0]
                    # gt_unit_positions = np.array(gt_obs["units"]["position"][target_team_id])  # (max_units, 2)
                    # gt_unit_energies = np.array(gt_obs["units"]["energy"][target_team_id]).flatten()  # (max_units,)
                    # assert np.all(gt_unit_positions == episode_store.own_unit_positions)
                    # assert np.all(gt_unit_energies == episode_store.own_unit_energies)  # gtの値が負の大きい値が出る

                    # hidden_state = extract_hidden_state(gt_obs, target_team_id)
                    # episode_hidden_state_group.create_dataset(f"{step_idx}", data=hidden_state)

                    # stateの次のステップにおけるactionを予測したいのでnext_stepの行動を取得する
                    next_actions = next_step_info[target_team_id]["action"]
                    action = extract_action(next_actions, obs, target_team_id)
                    episode_action_group.create_dataset(f"{step_idx}", data=action)

                max_steps.append(len(steps) - 1)
        return pl.DataFrame(
            {"EpisodeId": valid_ids, "MaxStep": max_steps},
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
