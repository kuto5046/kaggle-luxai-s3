import gzip
import json
from pathlib import Path

from joblib import Parallel, delayed


def compress_single_file(json_file, output_dir):
    # Read original json
    with open(json_file) as f:
        data = json.load(f)

    # Write compressed gzip
    output_path = output_dir / (json_file.stem + ".json.gz")
    with gzip.open(output_path, "wt") as f:
        json.dump(data, f)

    # Remove original json file
    json_file.unlink()


def compress_json_files():
    input_dir = Path("/home/task/kaggle/kaggle-luxai-s3/output/feature_store/episodes/42704976_raw")
    output_dir = Path("/home/task/kaggle/kaggle-luxai-s3/output/feature_store/episodes/42704976")
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    # Get all json files
    json_files = list(input_dir.glob("*.json"))

    # Process files in parallel
    Parallel(n_jobs=-1)(delayed(compress_single_file)(json_file, output_dir) for json_file in json_files)


if __name__ == "__main__":
    compress_json_files()
