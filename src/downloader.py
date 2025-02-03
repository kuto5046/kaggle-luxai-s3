import json
import time
import datetime
from pathlib import Path

import polars as pl
import requests
from tqdm.auto import tqdm

BASE_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/"
GET_URL = BASE_URL + "GetEpisodeReplay"


def saveEpisode(epid: int, save_path: Path) -> None:
    # request
    re = requests.post(GET_URL, json={"episodeId": int(epid)}, timeout=10)

    # save replay
    replay = re.json()
    with open(save_path, "w") as f:
        json.dump(replay, f)


def main():
    df = pl.read_csv("/home/user/work/output/feature_store/episodes/episodes0202.csv")
    start_time = datetime.datetime.now(tz=datetime.timezone.utc)
    episode_count = 0
    for _sub_id, df in df.group_by("SubmissionId"):
        sub_id = _sub_id[0]
        output_dir = Path(f"/home/user/work/output/feature_store/episodes/{sub_id}")
        output_dir.mkdir(exist_ok=True, parents=True)
        ep_ids = df["EpisodeId"].unique()
        for epid in tqdm(ep_ids):
            save_path = output_dir / f"{epid}.json"
            if save_path.exists():
                print(f"  file {epid}.json already exists")
                continue

            saveEpisode(epid, save_path)
            episode_count += 1

            # process 1 episode/sec
            spend_seconds = (datetime.datetime.now(tz=datetime.timezone.utc) - start_time).seconds
            if episode_count > spend_seconds:
                time.sleep(episode_count - spend_seconds)

        print(f"Episodes saved: {episode_count}")


if __name__ == "__main__":
    main()
