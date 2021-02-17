import argparse
import json
from typing import List


def merge_predictions(
    save_f: str, filenames: List[str], enforce_unique: bool = True
) -> None:
    r"""Utility to merge multiple predictions files generated from running
    inference. Useful for submitting predictins to the leaderboard where
    different models are used for different languages.
    """
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
        "filenames",
        nargs="+",
        type=str,
        help="names of predictions files to merge separated by spaces",
    )
    args = parser.parse_args()
    merge_predictions(args.saveas, args.filenames)


if __name__ == "__main__":
    main()
