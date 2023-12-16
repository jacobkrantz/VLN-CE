# Waypoint Models for Instruction-guided Navigation in Continuous Environments

<p align="center">
  <img width="800" height="299" src="/data/res/waypoint_example.gif" alt="Waypoint example GIF">
</p>

[Project Webpage](https://jacobkrantz.github.io/waypoint-vlnce/) — [Paper](https://arxiv.org/abs/2110.02207)

These config files exist to train and evaluate waypoint-based models for VLN-CE as published in ICCV 2021. They explore a spectrum of waypoint action spaces ranging from heading prediction to continuous range-limited coordinate prediction. Each model in the paper (Table 1) can be reproduced following the configs below:

| Row (Table 1) | Model | Dist. | Offset |   val_seen SR   |   val_unseen SR   |           Config           |
| :-----------: | :---: | :---: | :----: | :-------------: | :---------------: | :------------------------: |
|       1       |  WPN  |   C   |   C    |      0.40       |       0.34        | [1-wpn-cc.yaml](1-wpn-cc.yaml) |
|       2       |  WPN  |   D   |   C    |      0.38       |       0.36        | [2-wpn-dc.yaml](2-wpn-dc.yaml) |
|       3       |  WPN  |   D   |   D    |      0.35       |       0.28        | [3-wpn-dd.yaml](3-wpn-dd.yaml) |
|       4       |  WPN  |   D   |   -    |      0.39       |       0.31        | [4-wpn-d_.yaml](4-wpn-d_.yaml) |
|       5       |  HPN  |   -   |   C    |    **0.47**     |     **0.38**      | [5-hpn-_c.yaml](5-hpn-_c.yaml) |
|       6       |  HPN  |   -   |   -    |      0.44       |       0.34        | [6-hpn-__.yaml](6-hpn-__.yaml) |

|     |            Legend           |
| :-: | :-------------------------: |
| WPN | Waypoint Prediction Network |
| HPN | Heading Prediction Network  |
|  C  | Continuous                  |
|  D  | Discrete                    |

Models were trained via DDPPO using 64 GPUs. An example [slurm script](/sbatch_scripts/waypoint_train.sh).

## Pretrained Models

Pretrained weights for the best [WPN model](https://drive.google.com/file/d/1XaJbkPYsVZGoM2pyJJ9umeuQ1u8kl9Fm/view?usp=sharing) (row 2) and the best [HPN model](https://drive.google.com/file/d/1W_q1cqP7g6Y6jHaXKKyFDLpaE3pdRJnI/view?usp=sharing) (row 5):

```bash
# WPN.pth (97MB)
gdown https://drive.google.com/uc?id=1XaJbkPYsVZGoM2pyJJ9umeuQ1u8kl9Fm
# HPN.pth (97MB)
gdown https://drive.google.com/uc?id=1W_q1cqP7g6Y6jHaXKKyFDLpaE3pdRJnI
```

All Table 1 models, including best weights when paired with the discrete navigator (DN), can be downloaded here: [waypoint_weights.zip](https://drive.google.com/file/d/1pU5pJ8mpFv_TuITMIQugC52Q1ls6EjqU/view?usp=sharing) (788MB). Naming convention: `{row}-{WPN|HPN}-{c|d|_}-{c|d|_}-{discretenav|}.pth`.

All model weights are subject to the [Matterport3D Terms-of-Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).

### Evaluation

Below is an evaluation script for the WPN model with the continuous navigator (CN). All models were trained with sliding turned off and evaluated with sliding turned on. *Runtime: ~10 minutes.*

```bash
python run.py \
  --run-type eval \
  --exp-config vlnce_baselines/config/r2r_waypoint/2-wpn-dc.yaml \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  NUM_ENVIRONMENTS 8 \
  EVAL_CKPT_PATH_DIR data/checkpoints/pretrained/WPN.pth \
  RESULTS_DIR data/checkpoints/pretrained/WPN_CN_evals \
  EVAL.SPLIT val_unseen \
  EVAL.SAMPLE False \
  EVAL.USE_CKPT_CONFIG False
```

Models can also be evaluated with a discrete navigator (DN) as such:

```bash
cfg="vlnce_baselines/config/r2r_waypoint/2-wpn-dc.yaml"
cfg="$cfg,habitat_extensions/config/vlnce_waypoint_DN.yaml"

python run.py \
  --run-type eval \
  --exp-config $cfg \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  NUM_ENVIRONMENTS 8 \
  EVAL_CKPT_PATH_DIR data/checkpoints/pretrained/WPN.pth \
  RESULTS_DIR data/checkpoints/pretrained/WPN_DN_evals \
  EVAL.SPLIT val_unseen \
  EVAL.SAMPLE False \
  EVAL.USE_CKPT_CONFIG False
```

## Citing

If you use these waypoint models in your research, please cite the following [paper](https://arxiv.org/abs/2110.02207):

```tex
@inproceedings{krantz2021waypoint,
  title={Waypoint Models for Instruction-guided Navigation in Continuous Environments},
  author={Jacob Krantz and Aaron Gokaslan and Dhruv Batra and Stefan Lee and Oleksandr Maksymets},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
