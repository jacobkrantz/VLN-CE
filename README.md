# Room-Across-Room Habitat Challenge (RxR-Habitat)

Official starter code for the RxR-Habitat Challenge hosted at the **2021 CVPR Embodied AI Workshop**!

The RxR-Habitat Challenge tasks agents with a performing visual navigation by natural language instructions in novel environments. The challenge uses the new Room-Across-Room ([RxR](https://ai.google.com/research/rxr/)) dataset which:

- contains multilingual instructions (English, Hindi, Telugu),
- is an order of magnitude larger than existing datasets, and
- uses varied paths to break a shortest-path-to-goal assumption.

For this challenge, the RxR dataset has been ported to [continuous environments](https://jacobkrantz.github.io/vlnce/) for use with the [Habitat Simulator](https://aihabitat.org/). We hope the RxR-Habitat Challenge spurs progress in vision-and-language navigation, focusing on continuous environments that mimic the real world.

<p align="center">
  <img width="573" height="360" src="./data/res/rxr_teaser.gif" alt="VLN-CE comparison to VLN">
</p>

The RxR-Habitat Challenge is hosted by Oregon State University, Google Research, and Facebook AI Research.

## Links

- Challenge webpage: [ai.google.com/research/rxr/habitat](https://ai.google.com/research/rxr/habitat)
- Workshop webpage: [embodied-ai.org](https://embodied-ai.org/)

---

## Table of contents

1. [Timeline](#timeline)
1. [Setup](#setup)
1. [Leaderboard Submission](#leaderboard-submission)
1. [Citing RxR-Habitat Challenge](#citing-rxr-habitat-challenge)
1. [Starter Code](#starter-code)
1. [Questions](#questions)

## Timeline

|               Event               |       Date      |
|:---------------------------------:|:---------------:|
|          Challenge Launch         |   Feb 17, 2021  |
|          Leaderboard Open         |   Mar 1, 2021   |
|         Leaderboard Closes        |   May 31, 2021  |
| Workshop and Winners Announcement | Jun 19-25, 2021 |

## Setup

The starter code uses Python 3.7 and Habitat Simulator v0.1.7. We recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/), though it isn't necessary. If you are using conda, Habitat-Sim can easily be installed with:

```bash
conda install -c aihabitat -c conda-forge habitat-sim headless
```

Otherwise, follow the Habitat-Sim [installation instructions](https://github.com/facebookresearch/habitat-sim#installation). Then install Habitat-Lab version `0.1.7`:

```bash
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

Now you can clone this repository and install the rest of the dependencies:

```bash
git clone --branch rxr-habitat-challenge git@github.com:jacobkrantz/VLN-CE.git
cd VLN-CE
python -m pip install -r requirements.txt
```

### Matterport3D

The RxR dataset uses Matterport3D (MP3D) scene reconstructions. The official Matterport3D download script (`download_mp.py`) can be accessed by following the "Dataset Download" instructions on their [project webpage](https://niessner.github.io/Matterport/). The Habitat scene data is needed:

```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Extract this data to `data/scene_datasets/mp3d` such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 total scenes.

### Data

Download: [RxR_VLNCE_v0.zip](https://storage.googleapis.com/rxr-habitat/RxR_VLNCE_v0.zip)

The RxR-Habitat Challenge uses the [Room-Across-Room dataset](https://github.com/google-research-datasets/RxR) ported to continuous environments. The porting process follows the same steps used to port Room-to-Room (R2R), with details in [this paper](https://arxiv.org/abs/2004.02857). The dataset has train, val_seen, and val_unseen splits with both Guide and Follower trajectories ported. To use the baseline models with these splits, the precomputed BERT instruction features should be downloaded from [here](https://github.com/google-research-datasets/RxR#downloading-bert-text-features) and be saved or linked to `data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{instruction_id}_{language}_text_features.npz`.


The starter code expects these files to be in this structure:

```text
data
|-- RxR_VLNCE_v0
|   | -- train
|   |    |-- train_guide.json.gz
|   |    |-- train_guide_gt.json.gz
|   |    |-- train_follower.json.gz
|   |    |-- train_follower_gt.json.gz
|   | -- val_seen
|   |    |-- val_seen_guide.json.gz
|   |    |-- val_seen_guide_gt.json.gz
|   |    |-- val_seen_follower.json.gz
|   |    |-- val_seen_follower_gt.json.gz
|   | -- val_unseen
|   |    |-- val_unseen_guide.json.gz
|   |    |-- val_unseen_guide_gt.json.gz
|   |    |-- val_unseen_follower.json.gz
|   |    |-- val_unseen_follower_gt.json.gz
|   | -- text_features
|   |    |-- ...
```

## Citing RxR-Habitat Challenge

To cite the challenge, please cite the following papers ([RxR](https://arxiv.org/abs/2010.07954) and [VLN-CE](https://arxiv.org/abs/2004.02857)):

```tex
@inproceedings{ku2020room,
  title={Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding},
  author={Ku, Alexander and Anderson, Peter and Patel, Roma and Ie, Eugene and Baldridge, Jason},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={4392--4412},
  year={2020}
}

@inproceedings{krantz_vlnce_2020,
  title={Beyond the Nav-Graph: Vision and Language Navigation in Continuous Environments},
  author={Jacob Krantz and Erik Wijmans and Arjun Majundar and Dhruv Batra and Stefan Lee},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
 }
```

## Leaderboard Submission

The leaderboard will be open March 1, 2021.

For guidelines and access to the leaderboard, please visit: [ai.google.com/research/rxr/habitat](https://ai.google.com/research/rxr/habitat)

## Starter Code

We provide starter code for two models (Seq2Seq and CMA) and two trainer classes (`dagger_trainer` and `recollect_trainer`).

The `dagger_trainer` is the standard trainer used for VLN-CE baselines and can train with teacher forcing or dataset aggregation (DAgger). This trainer saves trajectories to disk to avoid time spent in simulation. NEW: saving to disk can take up a lot of space. The `IL.DAGGER.lmdb_fp16` flag can save your features in a compressed sized and expand them to fp32 again when loaded.

The `recollect_trainer` performs teacher forcing using the ground truth trajectories provided in the dataset rather than a shortest path expert. Also, this trainer does not save episodes to disk, instead opting to recollect them in simulation.

The `run.py` script is how training and evaluation is done for all model configurations. Specify a configuration file and a run type as such:

```bash
python run.py \
  --exp-config path/to/experiment_config.yaml \
  --run-type {train | eval | inference}
```

For example, a CMA model can be trained with teacher forcing on all English training episodes using the `recollect_trainer` as such:

```bash
python run.py \
  --exp-config vlnce_baselines/config/rxr_configs/rxr_cma_en.yaml \
  --run-type train
```

For lists of modifiable configuration options, see the default [task config](habitat_extensions/config/default.py) and [experiment config](vlnce_baselines/config/default.py) files.

### Encoder Weights

To encode depth observation, baseline models use a ResNet pretrained on a PointGoal navigation task using [DDPPO](https://arxiv.org/abs/1911.00357). These depth weights can be downloaded [here](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines/rl/ddppo) (672M). Extract the contents of `ddppo-models.zip` to `data/ddppo-models/{model}.pth`.

### Evaluating Models

Evaluation on validation splits can be done by running `python run.py --exp-config path/to/experiment_config.yaml --run-type eval`. Please see `EVAL` in the [default experiment config](vlnce_baselines/config/default.py). If `EVAL.EPISODE_COUNT == -1`, all episodes will be evaluated. If `EVAL_CKPT_PATH_DIR` is a directory, each checkpoint will be evaluated one at a time.

### Cuda

Cuda will be used by default if it is available. When training on large portions of the dataset, multiple GPUs is favorable.

```yaml
SIMULATOR_GPU_IDS: [0]  # Each GPU runs NUM_ENVIRONMENTS environments
TORCH_GPU_ID: 0
NUM_ENVIRONMENTS: 1
```

## Questions?

Feel free to contact the challenge organizers with any questions, comments, or concerns. The corresponding organizer is Jacob Krantz (@jacobkrantz). You can also open an issue with `[RxR-Habitat]` in the title, which will also notify us.
