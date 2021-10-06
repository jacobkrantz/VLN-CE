import argparse
import json
from typing import List

import jsonlines


def merge_predictions(
    save_f: str, filenames: List[str], enforce_unique: bool = True
) -> None:
    """Utility to merge multiple predictions files generated from running
    inference. Useful for submitting predictins to the leaderboard where
    different models are used for different languages.
    """
    if filenames[0].endswith(".jsonl"):  # merging rxr predictions
        merged_predictions = []
        for fn in filenames:
            with jsonlines.open(fn) as reader:
                for episode in reader:
                    merged_predictions.append(episode)

        merged_predictions.sort(key=lambda x: x["instruction_id"])

        if enforce_unique:
            unique_ids = {ep["instruction_id"] for ep in merged_predictions}
            assert len(merged_predictions) == len(unique_ids)

        with jsonlines.open(save_f, mode="w") as writer:
            writer.write_all(merged_predictions)
    else:  # merging r2r predictions
        merged_predictions = {}
        for fn in filenames:
            with open(fn, "r") as f:
                new_predictions = json.load(f)
                overlapping_keys = set(merged_predictions.keys()) & set(
                    new_predictions.keys()
                )
                assert (
                    enforce_unique or len(overlapping_keys) == 0
                ), f"{fn} contains overlapping keys: {overlapping_keys}"
                merged_predictions.update(new_predictions)

        with open(save_f, "w") as f:
            json.dump(merged_predictions, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saveas",
        metavar="S",
        type=str,
        required=True,
        help="Name of the merged predictions file",
    )
    parser.add_argument(
        "--filenames",
        nargs="+",
        type=str,
        help="names of predictions files to merge separated by spaces",
    )
    args = parser.parse_args()
    merge_predictions(args.saveas, args.filenames)


if __name__ == "__main__":
    main()
