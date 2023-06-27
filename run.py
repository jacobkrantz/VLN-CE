#!/usr/bin/env python3

import argparse
import os
import random

import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
import gc
import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config
from vlnce_baselines.nonlearning_agents import (
    evaluate_agent,
    nonlearning_inference,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment. "
             "If this is a directory, run all yaml file contained in the dir.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    if os.path.isdir(args.exp_config):
        conf_parameter = args.exp_config
        if os.path.isdir(conf_parameter):
            print("Running several config files from:", conf_parameter)
            for file in sorted(os.listdir(conf_parameter)):
                if file.endswith(".yaml") or file.endswith(".yml"):
                    file_path = os.path.join(conf_parameter, file)
                    print("exp_config", file_path)
                    run_exp(exp_config=file_path, run_type=args.run_type, opts=args.opts)
                else:
                    print("Not a valid config file:", file)
    else:
        run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if not logdir:
        logdir = "logs"
    os.makedirs(logdir, exist_ok=True)
    config_file_root__name = logdir+"/"+exp_config.split("/")[-1].split(".")[0]
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    log_file = config_file_root__name + "_" + config.LOG_FILE
    logger.add_filehandler(log_file)
    logger.info(f"config: {config}")

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    if run_type == "eval":
        torch.backends.cudnn.deterministic = True
        if config.EVAL.EVAL_NONLEARNING:
            evaluate_agent(config)
            return

    if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
        nonlearning_inference(config)
        return

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()

    # avoids to write to all previous files if running in a loop
    logger.removeHandler(logger.handlers[-1])
    gc.collect()

if __name__ == "__main__":
    main()
