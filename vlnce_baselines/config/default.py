from typing import List, Optional, Union

from habitat.config.default import Config as CN

from habitat_extensions.config.default import get_extended_config as get_task_config

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME = "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.NUM_PROCESSES = 4
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.LOG_FILE = "train.log"
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.EPISODE_COUNT = 2
_C.EVAL.EVAL_NONLEARNING = False
_C.EVAL.NONLEARNING = CN()
_C.EVAL.NONLEARNING.AGENT = "RandomAgent"

# -----------------------------------------------------------------------------
# INFERENCE CONFIG
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "test"
_C.INFERENCE.USE_CKPT_CONFIG = True
_C.INFERENCE.CKPT_PATH = "data/checkpoints/CMA_PM_DA_Aug.pth"
_C.INFERENCE.PREDICTIONS_FILE = "predictions.json"
_C.INFERENCE.INFERENCE_NONLEARNING = False
_C.INFERENCE.NONLEARNING = CN()
_C.INFERENCE.NONLEARNING.AGENT = "RandomAgent"

# -----------------------------------------------------------------------------
# DAGGER ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.DAGGER = CN()
_C.DAGGER.LR = 2.5e-4
_C.DAGGER.ITERATIONS = 10
_C.DAGGER.EPOCHS = 4
_C.DAGGER.UPDATE_SIZE = 5000
_C.DAGGER.BATCH_SIZE = 5
_C.DAGGER.P = 0.75
_C.DAGGER.LMDB_MAP_SIZE = 1.0e12
# How often to commit the writes to the DB, less commits is
# better, but everything must be in memory until a commit happens/
_C.DAGGER.LMDB_COMMIT_FREQUENCY = 500
_C.DAGGER.USE_IW = True
# If True, load precomputed features directly from LMDB_FEATURES_DIR.
_C.DAGGER.PRELOAD_LMDB_FEATURES = False
_C.DAGGER.LMDB_FEATURES_DIR = "data/trajectories_dirs/debug/trajectories.lmdb"
# load an already trained model for fine tuning
_C.DAGGER.LOAD_FROM_CKPT = False
_C.DAGGER.CKPT_TO_LOAD = "data/checkpoints/ckpt.0.pth"
# -----------------------------------------------------------------------------
# MODELING CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# on GT trajectories in the training set
_C.MODEL.inflection_weight_coef = 3.2

_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_instruction = False

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.max_length = 200
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 50
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = True
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = False

_C.MODEL.RGB_ENCODER = CN()
# 'SimpleRGBCNN' or 'TorchVisionResNet50'
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.output_size = 256

_C.MODEL.DEPTH_ENCODER = CN()
# 'VlnResnetDepthEncoder' or 'SimpleDepthCNN'
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 128
# type of resnet to use
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"
# path to DDPPO resnet weights
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = "data/ddppo-models/gibson-2plus-resnet50.pth"

_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type = "GRU"

_C.MODEL.SEQ2SEQ = CN()
_C.MODEL.SEQ2SEQ.use_prev_action = False

_C.MODEL.CMA = CN()
_C.MODEL.CMA.use = False
# Use the state encoding model in RCM. If false,
# will just concat inputs and run an RNN over them
_C.MODEL.CMA.rcm_state_encoder = False

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = False
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0  # loss multiplier


def get_config(
    config_paths: Optional[Union[List[str], str]] = None, opts: Optional[list] = None
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if config.BASE_TASK_CONFIG_PATH != "":
        config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
