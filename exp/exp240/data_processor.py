import json
import shutil
import logging
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass

import h5py
import numpy as np
import joblib
import polars as pl
from lightning import seed_everything
from lux.utils import (
    EpisodeStore,
    extract_state,
    extract_action,
    extract_gt_state,
    extract_global_state,
    extract_hidden_state,
    extract_hidden_global_state,
)
from tqdm.auto import tqdm
from lux.params import EnvParams
from sklearn.model_selection import KFold

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    exp_name: str = Path(__file__).parent.name
    seed: int = 2025
    debug: bool = False
    use_gt: bool = False
    n_splits: int = 5
    root_dir: Path = Path("/home/user/work")
    input_dir: Path = root_dir / "input"
    episode_dir: Path = root_dir / "output/feature_store/luxai-s3-frog-parade-42704976/episodes"
    episode_path: Path = episode_dir / "episodes.csv"
    feature_dir: Path = root_dir / f"output/feature_store/{exp_name}"
    target_team_name: str = "Frog Parade"
    target_sub_ids: list[int] = field(default_factory=lambda: [42704976])
    validation: bool = False


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
    for r in json_load["rewards"]:
        if r is None:
            print(f"rewards include None -> {json_load['rewards']}")
            return False
    win_idx = np.argmax([r or 0 for r in json_load["rewards"]])  # win or tie
    win_team = json_load["info"]["TeamNames"][win_idx]
    return win_team == target_team_name
    # return True


class DataProcessor:
    def __init__(self, cfg: Config) -> None:
        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.cfg = cfg
        self.episode_path = cfg.episode_path
        self.episode_dir = cfg.episode_dir
        self.feature_dir = cfg.feature_dir
        if self.feature_dir.exists():
            shutil.rmtree(self.feature_dir)
            print(f"remove {self.feature_dir}")
        self.feature_dir.mkdir(parents=True, exist_ok=True)

    def read_data(self) -> pl.DataFrame:
        episode_df = pl.read_csv(self.episode_path)
        episode_df = episode_df.filter(pl.col("SubmissionId").is_in(self.cfg.target_sub_ids))
        print(f"episode_df: {len(episode_df)}")
        # なぜかepisodeに重複があるため除去
        # episode_df = episode_df.sample(n=1000, seed=self.cfg.seed)
        episode_df = episode_df.unique("EpisodeId")
        print(f"unique episode_df: {len(episode_df)}")
        if self.cfg.debug:
            episode_df = episode_df.sample(n=5, seed=self.cfg.seed)
            # episode_df = episode_df.filter(pl.col("EpisodeId") == 67293512)
        return episode_df

    def _process_episode(self, row) -> tuple[str, int, int]:
        sub_id = row["SubmissionId"]
        episode_id = row["EpisodeId"]
        episode_path = self.episode_dir / f"{sub_id}/{episode_id}.json"

        try:
            with open(episode_path) as f:
                json_load = json.load(f)
        except json.JSONDecodeError as e:
            print(f"EpisodeId {episode_id}: {e}")
            return None

        # 無効なepisodeはスキップ(valueも学習したいのでskip)
        if not valid_episode(json_load, self.cfg.target_team_name):
            return None

        with h5py.File(self.feature_dir / f"temp_{episode_id}.h5", "w") as out_f:
            episode_group = out_f.create_group(f"{episode_id}")
            episode_action_group = episode_group.create_group("actions")
            episode_state_group = episode_group.create_group("states")
            episode_global_state_group = episode_group.create_group("global_states")
            episode_hidden_state_group = episode_group.create_group("hidden_states")
            episode_hidden_global_state_group = episode_group.create_group("hidden_global_states")
            episode_win_group = episode_group.create_group("win")

            target_team_id = np.argmax(json_load["rewards"])  # win or tie
            match_results = get_match_results(json_load, target_team_id)

            # episode内で獲得する情報
            env_params = EnvParams(**json_load["configuration"]["env_cfg"])
            episode_store = EpisodeStore(target_team_id, env_params, self.cfg.validation, episode_id)
            steps = json_load["steps"]
            gt_env_params = EnvParams(**steps[0][0]["info"]["replay"]["params"])
            for step_idx in range(len(steps) - 1):  # 505でdoneとなるため-1
                step_info = steps[step_idx]
                next_step_info = steps[step_idx + 1]
                obs = json.loads(step_info[target_team_id]["observation"]["obs"])
                gt_obs = step_info[0]["info"]["replay"]["observations"][0]

                # マッチごとにリセットされる要素をリセット
                if obs["match_steps"] == 0:
                    episode_store.reset()
                # リセット時以外はupdateをする
                else:
                    prev_actions = np.array(step_info[target_team_id]["action"])
                    episode_store.update(obs, prev_actions)

                if self.cfg.use_gt:
                    state = extract_gt_state(gt_obs, target_team_id)
                else:
                    state = extract_state(obs, target_team_id, episode_store)
                episode_state_group.create_dataset(f"{step_idx}", data=state)

                global_state = extract_global_state(obs, target_team_id, env_params, episode_store)
                episode_global_state_group.create_dataset(f"{step_idx}", data=global_state)

                hidden_state = extract_hidden_state(gt_obs, target_team_id)
                episode_hidden_state_group.create_dataset(f"{step_idx}", data=hidden_state)

                hidden_global_state = extract_hidden_global_state(gt_env_params)
                episode_hidden_global_state_group.create_dataset(f"{step_idx}", data=hidden_global_state)

                next_actions = next_step_info[target_team_id]["action"]
                action = extract_action(next_actions, obs, target_team_id)
                episode_action_group.create_dataset(f"{step_idx}", data=action)

                match_idx = obs["steps"] // (EnvParams.max_steps_in_match + 1)
                is_win = match_results[match_idx]
                episode_win_group.create_dataset(f"{step_idx}", data=is_win)

        return str(episode_id), len(steps) - 1, target_team_id, is_win

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        # 並列処理の実行
        n_jobs = joblib.cpu_count() if not self.cfg.debug else 1
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self._process_episode)(row) for row in tqdm(df.iter_rows(named=True), total=len(df))
        )

        # 有効なエピソードのみを抽出
        valid_results = [r for r in results if r is not None]
        valid_ids, max_steps, target_team_ids, is_wins = zip(*valid_results)

        # 一時ファイルを1つのh5ファイルにマージ
        with h5py.File(self.feature_dir / "episodes.h5", "w") as out_f:
            for episode_id in valid_ids:
                temp_path = self.feature_dir / f"temp_{episode_id}.h5"
                with h5py.File(temp_path, "r") as temp_f:
                    temp_f.copy(f"{episode_id}", out_f)
                temp_path.unlink()  # 一時ファイルの削除

        return pl.DataFrame(
            {"EpisodeId": valid_ids, "MaxStep": max_steps, "TargetTeamId": target_team_ids, "Win": is_wins},
        )

    def add_fold(self, df: pl.DataFrame) -> pl.DataFrame:
        return get_kfold(df, self.cfg.n_splits, self.cfg.seed)

    def run(self) -> None:
        episode_paths = self.read_data()
        df = self.preprocess(episode_paths)
        if not self.cfg.debug:
            df = self.add_fold(df)
        df.write_csv(self.feature_dir / "train.csv")


def get_match_results(json_load: dict[str, Any], target_team_id: int) -> list[bool]:
    match_results = []
    for i_match in range(EnvParams.match_count_per_episode):
        final_step_in_match = (i_match + 1) * 100 + i_match  # 100, 201, 302, 403, 504

        teams_wins_after = np.asarray(
            json_load["steps"][final_step_in_match + 1][0]["info"]["replay"]["observations"][0]["team_wins"]
        )
        teams_wins_before = np.asarray(
            json_load["steps"][final_step_in_match][0]["info"]["replay"]["observations"][0]["team_wins"]
        )
        win_team = np.argmax(teams_wins_after - teams_wins_before)

        is_win = win_team == target_team_id
        match_results.append(is_win)
    return match_results


def main() -> None:
    cfg = Config()
    data_processor = DataProcessor(cfg)
    # data_processor.test()
    data_processor.run()


if __name__ == "__main__":
    main()
