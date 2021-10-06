import argparse
import copy

import torch


def ckpt_to_interrupted_state(ckpt: str, save_f: str) -> None:
    """A checkpoint saved with the necessary data can be converted to
    an interrupted_state for requeuing the SLURM job.
    """
    c = torch.load(ckpt)
    state = {
        "state_dict": copy.deepcopy(c["state_dict"]),
        "optim_state": copy.deepcopy(c["extra_state"]["optim_state"]),
        "lr_sched_state": copy.deepcopy(c["extra_state"]["lr_sched_state"]),
        "config": copy.deepcopy(c["config"]),
        "requeue_stats": copy.deepcopy(c["extra_state"]["requeue_stats"]),
    }
    torch.save(state, save_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a checkpoint to an interrupted state"
    )
    parser.add_argument(
        "--ckpt",
        metavar="c",
        type=str,
        required=True,
        help="Checkpoint to be converted",
    )
    parser.add_argument(
        "--saveas",
        metavar="s",
        type=str,
        required=False,
        default="data/interrupted_state.pth",
        help="file to save the new interrupted state to",
    )
    args = parser.parse_args()
    ckpt_to_interrupted_state(ckpt=args.ckpt, save_f=args.saveas)
