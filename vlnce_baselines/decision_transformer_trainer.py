import gc
import random
import warnings

import lmdb
import msgpack_numpy
import numpy as np
from vlnce_baselines.common.base_il_trainer import BaseILTrainer
from vlnce_baselines.common.env_utils import construct_envs

from torch import Tensor
import re
import json
import os
import time
import warnings
from collections import defaultdict
import torch
import torch.nn.functional as F
import tqdm
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.common import batch_obs

from habitat_extensions.utils import generate_video, observations_to_image
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401
import jsonlines
from typing import Any, Dict, List, Optional, Tuple


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


# Trick to create extra start token directly in the collate_fn
# we don t need to recreate the whole dataset...
global EXTRA_START_TOKEN_ID
global STOP_ACTION_TOKEN_ID
EXTRA_START_TOKEN_ID = 4
STOP_ACTION_TOKEN_ID = 0


def _is_correct_previous_actions(batch):
    """
    Somehow, I detected a bug. Some actions are not shifted correctly
    prev_actions_batch[i+1] is not always equal to  corrected_actions_batch[i]
    :param batch:
    :return:
    """
    return sum([(batch[i][1][1:] == batch[i][2][:-1]).sum() == len(batch[i][1][1:]) for i in range(len(batch))]) == len(
        batch)


def collate_fn_check_batch(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """
    if not _is_correct_previous_actions(batch):
        raise Exception(
            "Dataset has not been created correctly! Prev actions and corrected actions not shifted accordingly!")


def _block_shuffle(lst, block_size):
    blocks = [lst[i: i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
        preload_size=128
    ):
        super().__init__()
        assert preload_size > 0
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * preload_size
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=True
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break
                    entry = txn.get(str(self.load_ordering.pop()).encode())
                    unpacked = msgpack_numpy.unpackb(entry,raw=False,)
                    new_preload.append(unpacked)

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self


@baseline_registry.register_trainer(name="decision_transformer")
class DecisionTransformerTrainer(BaseILTrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        #
        self.rewards = {"point_nav_reward": {
            "step_penalty": config.IL.DECISION_TRANSFORMER.POINT_GOAL_NAV_REWARD.step_penalty,
            "success": config.IL.DECISION_TRANSFORMER.POINT_GOAL_NAV_REWARD.success},
            "sparse_reward": {
                "step_penalty": config.IL.DECISION_TRANSFORMER.SPARSE_REWARD.step_penalty,
                "success": config.IL.DECISION_TRANSFORMER.SPARSE_REWARD.success},
            "ndtw_reward": {
                "step_penalty": config.IL.DECISION_TRANSFORMER.NDTW_REWARD.step_penalty,
                "success": config.IL.DECISION_TRANSFORMER.NDTW_REWARD.success},
        }
        device = "cuda:" + str(config.TORCH_GPU_ID)

        self.rgb_depth_stats = {
            "mean_rgb": torch.as_tensor(np.asarray([0.533, 0.498, 0.453]),
                                        dtype=torch.float32, device=device),
            "std_rgb": torch.as_tensor(np.asarray([0.183, 0.185, 0.2020]),
                                       dtype=torch.float32, device=device),
            "mean_depth": torch.as_tensor(np.asarray([0.222]), dtype=torch.float32, device=device),
            "std_depth": torch.as_tensor(np.asarray([0.18]), dtype=torch.float32, device=device)}



        super().__init__(config)

    def _create_feature_hooks(self):
        self.rgb_features = None
        self.rgb_hook = None

        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.cpu())

            return hook

        if not self.config.MODEL.RGB_ENCODER.trainable:
            self.rgb_features = torch.zeros((1,), device="cpu")
            self.rgb_hook = self.policy.net.rgb_encoder.cnn.register_forward_hook(
                hook_builder(self.rgb_features)
            )

        self.depth_features = None
        self.depth_hook = None
        if not self.config.MODEL.DEPTH_ENCODER.trainable:
            self.depth_features = torch.zeros((1,), device="cpu")
            self.depth_hook = self.policy.net.depth_encoder.visual_encoder.register_forward_hook(
                hook_builder(self.depth_features)
            )

    def _release_hook(self):

        if self.rgb_hook is not None:
            self.rgb_hook.remove()
        if self.depth_hook is not None:
            self.depth_hook.remove()
        self.rgb_features = torch.zeros((1,), device="cpu")
        self.depth_features = torch.zeros((1,), device="cpu")

    def _calculate_return_to_go(self, traj_obs: dict, reward_type: str, observation_type: str, scaling_factor=1.0,
                                destination_key=None):
        """
        Calculate the return to go. For a given step, sum of all rewards to come
        :param traj_obs:
        :param reward_type:
        :param observation_type:
        :param destination_key: if not given, will try to derive a destination name from observation_type (should start with raw_)
        :param scaling_factor: scale the rewards down, proportinally to the sequence length
        :return:
        """
        assert (reward_type in self.rewards.keys())
        assert (observation_type in traj_obs.keys())
        rewards = traj_obs[observation_type]
        # work around when transforming the values read in the database...
        # that avoid to have a second fonction for the collate_fn when recalculating on the fly
        isTensor = type(rewards) is torch.Tensor
        if isTensor:
            rewards = rewards.numpy()
        rewards = rewards + self.rewards[reward_type]["step_penalty"]
        rewards[-1] = rewards[-1] + self.rewards[reward_type]["success"]
        # Just save the simple rewards for each time steps, not accumulated if needed
        rewards = np.float32(rewards.squeeze())
        # In some cases, when using dagger, The agent stops on first action. Hence,
        # we need to transform the scalar in array...
        if type(rewards) is np.float32:
            rewards = np.array([rewards])
        if isTensor:
            simple_reward = torch.from_numpy(rewards)
        else:
            simple_reward = rewards
        traj_obs[reward_type] = simple_reward
        rewards = np.flip(np.flip(rewards).cumsum())
        if destination_key is None:
            assert observation_type.startswith("raw_")
            destination_key = observation_type.split("raw_")[1] + "_to_go"
        reward_to_go = np.float32(rewards / scaling_factor)
        if isTensor:
            reward_to_go = torch.from_numpy(reward_to_go)
        traj_obs[destination_key] = reward_to_go

    def _calculate_rewards(self, traj_obs, scaling_factor):
        self._calculate_return_to_go(traj_obs, "point_nav_reward", "raw_point_nav_reward", scaling_factor)
        self._calculate_return_to_go(traj_obs, "sparse_reward", "raw_sparse_reward", scaling_factor)
        self._calculate_return_to_go(traj_obs, "ndtw_reward", "raw_ndtw_reward", scaling_factor)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def _prepare_observation(self, observations):
        '''
        From the observation created by the environment, creates dictionaries of features,
        with shape : number of environment * all the remaning features.
        :param observations:
        :return:
        '''
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        return observations, batch

    def _modify_batch_for_transformer(self, episodes: list, batch: ObservationsDict, rgb_features: Tensor,
                                      depth_features: Tensor, envs, prev_actions, rgb_key, depth_key):
        """
        This function help to prepare the input needed by the transformer model.
        Habitat Sim provides one image / observation per time step, we need the whole serie for the transformer model.
        Hence, the batch parameter is modified to have a whole sequence of rgb, depth and instructions
        Moreover, the previous actions are returned by this function (only important at the first timestep)
        :param episodes:
        :param batch:
        :param rgb_features:
        :param depth_features:
        :param envs:
        :param prev_actions:
        :param rgb_key:
        :param depth_key:
        :return:
        """
        current_rgb = rgb_features.unsqueeze(dim=1)
        current_depth = depth_features.unsqueeze(dim=1)

        if not self._are_episodes_empty(episodes):
            # preparing the past images as a sequence
            rgb_seq = self._create_sequence(episodes, rgb_key)
            depth_seq = self._create_sequence(episodes, depth_key)
            # adding the current image to the end of the sequence
            rgb_seq = torch.cat((rgb_seq, current_rgb), dim=1)
            depth_seq = torch.cat((depth_seq, current_depth), dim=1)
        else:
            # we unsqueeze at dim = 1 to create a shape of of batch, sequence (of size 1 at the beginning), and all other dim
            rgb_seq = current_rgb
            depth_seq = current_depth
            prev_actions = torch.zeros(
                envs.num_envs,
                1,
                device=self.device,
                dtype=torch.long,
            )
            if self.config.MODEL.DECISION_TRANSFORMER.use_extra_start_token:
                prev_actions = prev_actions + EXTRA_START_TOKEN_ID
        # store the last images
        for i in range(envs.num_envs):
            episodes[i].append({rgb_key: rgb_seq[i][-1], depth_key: depth_seq[i][-1]})
        seq_length = rgb_seq.shape[1]
        # just repeat the instructions for each time step
        batch["instruction"] = batch["instruction"].unsqueeze(dim=1).repeat(1, seq_length, 1)
        # setting it here to not trigger the hook another time and increase processing time
        batch[rgb_key] = rgb_seq.to(self.device)
        batch[depth_key] = depth_seq.to(self.device)

        return prev_actions

    def _create_sequence(self, episodes: list, feature_key: str):
        '''
        Returns a tensor corresponding to the sequence of features. It has a shape
        number of environments * sequence length * all the remaining dimensions
        :param episodes: list of active environments; each environment contains the sequence of observations
        :param feature_key: "rgb_features", "depth_features"
        :return:
        '''
        return torch.stack(
            [torch.stack([obs[feature_key] for obs in ep], dim=0) for env, ep in enumerate(episodes) if len(ep) > 0],
            dim=0)

    def _are_episodes_empty(self, episodes: list):
        '''
        Check if the current list of steps is empty or not.
        :param episodes:
        :return:
        '''
        return sum([len(e) > 0 for e in episodes]) == 0

    def _filter_envs_episodes(self,
        envs_to_pause,
        envs,
        episodes=None,

    ):
        """
        Same logic as _pause_envs in the BaseTrainer, but does not pause the episodes.
        Must be called before calling _pause_envs(), on lists that can fit in
        _pause_envs()
        Args:
            envs_to_pause:
            envs:
            episodes:

        Returns:

        """
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):#The envs to paused as to be done so in reverse order, otherwise, you mess up the index...
                state_index.pop(idx)
            if episodes is not None:
                episodes = [episodes[i] for i in state_index]
        return episodes
    def _update_dataset(self, data_it):
        """
        Cache the whole dataset. Data Aggregation can be used, the trained model
        can output some action for time steps at a given probability. As the whole Task rely on an Oracle that
        can output the best decision to reach the next trajectory node, even a bad decision of the model can be recovered
        (imagine backtracking...), hence effectively implementing DAGGER.
        :param data_it:
        :return:
        """
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid
        distance_left_uuid = self.config.IL.DECISION_TRANSFORMER.sensor_uuid
        hidden_states = torch.zeros(envs.num_envs, 1,
                                    dtype=torch.float)  # more of a placeholder, we don t need it for the transformer
        # prev_actions = torch.zeros(
        #     envs.num_envs,
        #     1,
        #     device=self.device,
        #     dtype=torch.long,
        # )
        prev_actions = None
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        observations = envs.reset()
        observations, batch = self._prepare_observation(observations)
        # initialize at dim 1 for sequences of frames etc...

        episodes = [[] for _ in range(envs.num_envs)]
        episode_features = [[] for _ in range(envs.num_envs)]

        skips = [False for _ in range(envs.num_envs)]
        # Populate dones with False initially
        dones = [False for _ in range(envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        p = self.config.IL.DAGGER.p
        # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
        beta = 0.0 if p == 0.0 else p ** data_it

        ensure_unique_episodes = beta == 1.0

        self._create_feature_hooks()

        # That needs to be turned to eval when doing Dagger! Not on the original implementation
        self.policy.eval()

        rgb_encoder = self.policy.net.rgb_encoder
        depth_encoder = self.policy.net.depth_encoder

        collected_eps = 0
        ep_ids_collected = []
        if ensure_unique_episodes:
            ep_ids_collected = set()

        dataset_episodes = sum(envs.number_of_episodes)
        print("Numbers of episodes in the split:", dataset_episodes)
        if (self.config.IL.DAGGER.update_size > dataset_episodes and ensure_unique_episodes):
            collect_size = dataset_episodes
            print("Ensure unique episodes")
        else:
            print("Unique episodes not enforced")
            collect_size = self.config.IL.DAGGER.update_size

        print(f"To be collected: {collect_size} ")
        horizon = 1
        agent_action = False

        def _detect_wrong_episode(transposed_ep):
            return not (transposed_ep[1][1:] == transposed_ep[2][:-1]).sum() == len(transposed_ep[1][1:])

        collected_eps_for_real = 0
        with tqdm.tqdm(
            total=collect_size, dynamic_ncols=True
        ) as pbar, lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        ) as lmdb_env, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)
            last_episodes = envs.current_episodes()
            while collected_eps < collect_size:
                envs_to_pause = []
                current_episodes = envs.current_episodes()
                # if the max steps of the transform model is reached,
                # and the agent does not call the stop action, force the agent to ignore the episode
                if horizon == self.config.IL.DECISION_TRANSFORMER.episode_horizon:
                    episode_end = torch.where(actions == STOP_ACTION_TOKEN_ID, True, False)
                    for i in range(envs.num_envs):
                        if not episode_end[i] and i not in envs_to_pause:
                            skips[i] = True
                            envs_to_pause.append(i)

                for i in range(envs.num_envs):

                    if dones[i] and not skips[i]:
                        ep = episodes[i]
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        )
                        del traj_obs[expert_uuid]
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()
                            if self.config.IL.DAGGER.lmdb_fp16:
                                traj_obs[k] = traj_obs[k].astype(np.float16)
                        # First step: calculate the difference between 2 consecutive time steps.
                        # We add the initial distance to the goal to calculate the first differential reward
                        # traj_obs["point_nav_reward_to_go"] = np.diff(
                        #    np.concatenate(([current_episodes[i].info["geodesic_distance"]], traj_obs[distance_left_uuid])), axis=0)
                        # We add the final distance to the goal once again because on th elast step,
                        # the STOP action is called

                        traj_obs["raw_point_nav_reward"] = np.diff(
                            np.concatenate((traj_obs[distance_left_uuid], [traj_obs[distance_left_uuid][-1]])),
                            axis=0) * -1.0
                        # PReparing entries for sparse rewards
                        traj_obs["raw_sparse_reward"] = np.zeros_like(traj_obs["raw_point_nav_reward"])
                        scaling_factor = traj_obs[distance_left_uuid].size  # Scaling by the episode length
                        del traj_obs[distance_left_uuid]
                        traj_obs["raw_ndtw_reward"] = np.array([step[3] for step in ep], dtype=np.float16)
                        self._calculate_rewards(traj_obs, scaling_factor)
                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]

                        is_episode_perfect = True

                        if self.config.IL.DECISION_TRANSFORMER.use_perfect_episode_only_for_dagger:
                            is_episode_perfect  = infos[i]["success"] == 1.0

                        # don t add anything that seems weird
                        if is_episode_perfect:
                            txn.put(
                                str(start_id + collected_eps_for_real).encode(),
                                msgpack_numpy.packb(
                                    transposed_ep, use_bin_type=True
                                ),
                            )
                            collected_eps_for_real += 1
                        # incrementinmg here outside the if block is not a bug
                        # If we can't add successfull episodes (while using dagger),
                        # we still need a way to exit the update function...
                        collected_eps += 1
                        pbar.update()


                        if (
                            collected_eps_for_real > 0 and collected_eps_for_real
                            % self.config.IL.DAGGER.lmdb_commit_frequency
                        ) == 0:
                            txn.commit()
                            txn = lmdb_env.begin(write=True)

                        if ensure_unique_episodes:
                            if (not last_episodes[i].episode_id in ep_ids_collected):
                                ep_ids_collected.add(current_episodes[i].episode_id)

                    # In opposition to the RNN logic, where only one state per time step is handled,
                    # We need this to force all sequences in the current batch to finish...
                    if dones[i]:
                        if i not in envs_to_pause:
                            envs_to_pause.append(i)

                envs_to_pause = sorted(envs_to_pause)

                episode_features = self._filter_envs_episodes(envs_to_pause, envs, episode_features)

                try:
                    (
                        envs,
                        hidden_states,
                        not_done_masks,
                        prev_actions,
                        batch,
                        episodes,
                    ) = self._pause_envs(
                        envs_to_pause,
                        envs,
                        hidden_states,
                        not_done_masks,
                        prev_actions,
                        batch,
                        episodes,  # A trick, I am using what is thought for the RGB features to reduce this list as well
                    )
                except Exception as e:
                    logger.warning(f"Something went wrong! Dagger It {data_it}")
                    for j in range(len(current_episodes)):
                        logger.warning(f"Current Episode culprit: {current_episodes[j].episode_id} , env {j}")
                    for j in range(len(last_episodes)):
                        logger.warning(f"Last Episode culprit: {last_episodes[j].episode_id} , env {j}")
                    logger.warning(f"Current horizon:{horizon}")
                    logger.warning(envs_to_pause)
                    logger.warning(f"Num envs :{envs.num_envs}")
                    save_file = f"ckpt.{data_it * self.config.IL.epochs}.pth"
                    self.save_checkpoint(
                        save_file
                    )
                    logger.warning(f"Saved : {save_file}")
                    raise e

                if envs.num_envs == 0:
                    envs.resume_all()
                    observations = envs.reset()
                    # This piece of code enforce to load only episode
                    # not previously collected.
                    to_init = min((collect_size - len(ep_ids_collected)), envs.num_envs)
                    if ensure_unique_episodes and to_init > 0:
                        initialized = 0
                        while initialized < to_init:
                            if initialized > 0:
                                initialized = 0
                            for env, e in enumerate(envs.current_episodes()):
                                if e.episode_id in ep_ids_collected:
                                    observations[env] = envs.reset_at(env)[0]
                                else:
                                    initialized += 1
                    current_episodes = envs.current_episodes()
                    episodes = [[] for _ in range(envs.num_envs)]
                    episode_features = [[] for _ in range(envs.num_envs)]
                    prev_actions = None
                    observations, batch = self._prepare_observation(observations)
                    self.rgb_features = self.rgb_features.set_(torch.zeros((1,), device="cpu"))
                    self.depth_features = self.depth_features.set_(torch.zeros((1,), device="cpu"))

                self._normalize_depth(batch)
                rgb_encoder(batch)
                depth_encoder(batch)
                for i in range(envs.num_envs):
                    if self.rgb_features is not None:
                        observations[i]["rgb_features"] = self.rgb_features[i]
                        del observations[i]["rgb"]

                    if self.depth_features is not None:
                        observations[i]["depth_features"] = self.depth_features[i]
                        del observations[i]["depth"]

                prev_actions = self._modify_batch_for_transformer(episode_features, batch, self.rgb_features,
                                                                  self.depth_features, envs,
                                                                  prev_actions,
                                                                  "rgb_features", "depth_features")
                batch_size = prev_actions.shape[0]
                horizon = prev_actions.shape[1]
                perform_dagger = (torch.rand((batch_size, 1), dtype=torch.float) < beta).to(self.device)
                # only perform dagger when the random process allows it (should lower the
                # processing time...)
                if perform_dagger.sum() < batch_size:
                    agent_action = True
                    actions, _ = self.policy.act(
                        batch,
                        hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                    )
                else:
                    actions = torch.ones_like(batch[expert_uuid].long())
                # actions.shape[0] == number of active enviroments
                hidden_states = torch.zeros(actions.shape[0], 1, dtype=torch.float)

                actions = torch.where(
                    perform_dagger,
                    batch[expert_uuid].long(),
                    actions,
                )

                if self.config.IL.DECISION_TRANSFORMER.use_oracle_actions:
                    next_actions = batch[expert_uuid] #This is maybe a big bug for Dagger. Because if we do that like this, the sequences won't be aligned anymore
                else:
                    next_actions = actions
                # We gathered images, actions, and sensor feedback for current timestep
                # time to save the timestep

                for i in range(envs.num_envs):
                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i, -1].item(),  # this is a sequence of actions, we take the last action
                            next_actions[i].item(),
                        )
                    )

                skips = batch[
                            expert_uuid].long() == -1  # looks like the short path sensor return -1 if there is a problem,, hence you need to skip an environment
                actions = torch.where(
                    skips, torch.zeros_like(actions), actions
                )
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)
                # add the last actions to the sequence of previous actions
                prev_actions = torch.cat([prev_actions, actions], dim=1).to(self.device)
                # prev_actions.copy_(actions)

                # When we step, environments can be reloaded automatically.
                # we need to cache the previous list of episodes to be able to add them correctly in the part
                # where done and not skip is applied.
                last_episodes = current_episodes
                outputs = envs.step([a[0].item() for a in actions])

                observations, _, dones, infos = [list(x) for x in zip(*outputs)]

                # Just add ndtw, if you need it as Reward
                for i in range(envs.num_envs):
                    obs, prev_act, next_act = episodes[i][-1]
                    episodes[i][-1] = (obs, prev_act, next_act, infos[i]["ndtw"])

                observations, batch = self._prepare_observation(observations)

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

            txn.commit()

        envs.close()
        envs = None

        self._release_hook()
        if agent_action:
            print("Dataset Creation with some agent actions.")
        # That needs to be turned back on...
        self.policy.train()

    def inference(
        self,
    ):
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        """
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.INFERENCE.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.INFERENCE.LANGUAGES
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = config.INFERENCE.CKPT_PATH
        config.TASK_CONFIG.TASK.MEASUREMENTS = []
        config.TASK_CONFIG.TASK.SENSORS = [
            s for s in config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s
        ]
        config.ENV_NAME = "VLNCEInferenceEnv"
        config.freeze()

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )

        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()

        self._create_feature_hooks()
        rgb_encoder = self.policy.net.rgb_encoder
        depth_encoder = self.policy.net.depth_encoder

        observations = envs.reset()
        observations, batch = self._prepare_observation(observations)

        prev_actions = None
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        episode_predictions = defaultdict(list)

        episodes = [[] for _ in range(envs.num_envs)]

        # episode ID --> instruction ID for rxr predictions format
        instruction_ids: Dict[str, int] = {}

        episode_already_predicted = []

        def _populate_episode_with_starting_states():
            # populate episode_predictions with the starting state
            current_episodes = envs.current_episodes()
            for i in range(envs.num_envs):
                ep_id = current_episodes[i].episode_id
                if ep_id not in episode_already_predicted:
                    episode_predictions[current_episodes[i].episode_id].append(
                        envs.call_at(i, "get_info", {"observations": {}})
                    )
                if config.INFERENCE.FORMAT == "rxr":
                    ep_id = current_episodes[i].episode_id
                    k = current_episodes[i].instruction.instruction_id
                    instruction_ids[ep_id] = int(k)

        _populate_episode_with_starting_states()

        num_eps = sum(envs.count_episodes())
        pbar = tqdm.tqdm(total=num_eps) if hasattr(config, "use_pbar") and config.use_pbar else None

        # if all envs finishes at the same time, the operation is equal to 1.
        # if all env are still processing, the operation is simply zero...
        has_env_finished_early = lambda envs_that_needs_to_wait: sum(envs_that_needs_to_wait.values()) / len(
            envs_that_needs_to_wait) > 0

        while envs.num_envs > 0 and len(episode_already_predicted) < num_eps:

            current_episodes = envs.current_episodes()
            # caching the outputs of the cnn on one image only
            self._normalize_depth(batch)
            rgb_encoder(batch)
            depth_encoder(batch)
            del batch["rgb"]
            del batch["depth"]
            rgb_key = "rgb_features"
            depth_key = "depth_features"
            prev_actions = self._modify_batch_for_transformer(episodes, batch, self.rgb_features, self.depth_features,
                                                              envs,
                                                              prev_actions, rgb_key, depth_key)

            with torch.no_grad():
                actions, hidden_states = self.policy.act(
                    batch,
                    None,
                    prev_actions,
                    not_done_masks,
                    deterministic=not config.EVAL.SAMPLE,
                )
                prev_actions = torch.cat([prev_actions, actions], dim=1).to(self.device)
                # prev_actions.copy_(actions)

            horizon = prev_actions.shape[1]
            # if the max steps of the transform model is reached, force to end the game
            if horizon == self.config.IL.DECISION_TRANSFORMER.episode_horizon:
                actions[:, -1] = 0

            outputs = envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            # need to use a deep copy, otherwise, observations would be the same as cleaned_observations
            observations, batch = self._prepare_observation(observations)
            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )

            # reset envs and observations if necessary
            envs_that_needs_to_wait = {}
            for i in range(envs.num_envs):
                ep_id = current_episodes[i].episode_id
                if ep_id not in episode_already_predicted:
                    episode_predictions[ep_id].append(infos[i])
                # This helps us to generate the transformer sequence
                # episodes[i].append((cleaned_observations[i], prev_actions[i, -1].item()))
                if not dones[i]:
                    envs_that_needs_to_wait[i] = False
                    continue
                envs_that_needs_to_wait[i] = True
                episodes[i] = []
                episode_already_predicted.append(ep_id)
                # observations[i] = envs.reset_at(i)[0]
                # This step is usually done in self._prepare_observation(observations)
                # but now, because we amenbd only one observation, we need to take care of this step manually...
                # observations[i][self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID] = \
                # observations[i][self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID]["tokens"]
                # self.rgb_features = self.rgb_features.set_(torch.zeros((1,), device="cpu"))
                # self.depth_features = self.depth_features.set_(torch.zeros((1,), device="cpu"))
                # observations, batch = self._prepare_observation(observations)

                if pbar:
                    pbar.update()

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in episode_already_predicted:
                    envs_to_pause.append(i)
                elif has_env_finished_early(envs_that_needs_to_wait):
                    if envs_that_needs_to_wait[i] and i not in envs_to_pause:
                        envs_to_pause.append(i)
            episodes = self._filter_envs_episodes(envs_to_pause, envs, episodes)
            (
                envs,
                hidden_states,
                not_done_masks,
                prev_actions,
                batch,
                _,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                hidden_states,
                not_done_masks,
                prev_actions,
                batch,
                None,
            )

            # at this stage, if we dont have any env left,
            # that means that all prediction within the same "batch"
            # are finished, we can wake all envs now.
            if envs.num_envs < 1:
                envs.resume_all()
                episodes = [[] for _ in range(envs.num_envs)]
                observations = envs.reset()
                prev_actions = None
                observations, batch = self._prepare_observation(observations)
                _populate_episode_with_starting_states()

        envs.close()
        gc.collect()
        self._release_hook()
        if pbar:
            pbar.close()

        if config.INFERENCE.FORMAT == "r2r":
            with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(episode_predictions, f, indent=2)

            logger.info(
                f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
            )
        else:  # use 'rxr' format for rxr-habitat leaderboard
            predictions_out = []

            for k, v in episode_predictions.items():

                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if path[-1] != p["position"]:
                        path.append(p["position"])

                predictions_out.append(
                    {
                        "instruction_id": instruction_ids[k],
                        "path": path,
                    }
                )

            predictions_out.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(
                config.INFERENCE.PREDICTIONS_FILE, mode="w"
            ) as writer:
                writer.write_all(predictions_out)

            logger.info(
                f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
            )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")

        config = self.config.clone()
        if self.config.EVAL.USE_CKPT_CONFIG:
            ckpt = self.load_checkpoint(checkpoint_path, map_location="cpu")
            config = self._setup_eval_config(ckpt)

        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            model_file = checkpoint_path.split("/")[-1]
            base_name = f"ckpt_{checkpoint_index}"
            if model_file != base_name:
                base_name = model_file
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_{base_name}_{split}.json",
            )
            if os.path.exists(fname):
                logger.info(f"skipping {base_name} -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()

        self._create_feature_hooks()
        rgb_encoder = self.policy.net.rgb_encoder
        depth_encoder = self.policy.net.depth_encoder

        observations = envs.reset()
        observations, batch = self._prepare_observation(observations)

        hidden_states = torch.zeros(envs.num_envs, 1, dtype=torch.float)

        prev_actions = None
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        episodes = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        # if all envs finishes at the same time, the operation is equal to 1.
        # if all env are still processing, the operation is simply zero...
        has_env_finished_early = lambda envs_that_needs_to_wait: sum(envs_that_needs_to_wait.values()) / len(
            envs_that_needs_to_wait) > 0

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()

            # caching the outputs of the cnn on one image only
            self._normalize_depth(batch)
            rgb_encoder(batch)
            depth_encoder(batch)
            del batch["rgb"]
            del batch["depth"]
            rgb_key = "rgb_features"
            depth_key = "depth_features"
            prev_actions = self._modify_batch_for_transformer(episodes, batch, self.rgb_features, self.depth_features,
                                                              envs,
                                                              prev_actions, rgb_key, depth_key)

            with torch.no_grad():
                actions, hidden_states = self.policy.act(
                    batch,
                    None,
                    prev_actions,
                    not_done_masks,
                    deterministic=not config.EVAL.SAMPLE,
                )
                prev_actions = torch.cat([prev_actions, actions], dim=1).to(self.device)
                # prev_actions.copy_(actions)

            horizon = prev_actions.shape[1]
            # if the max steps of the transform model is reached, force to end the game
            if horizon == self.config.IL.DECISION_TRANSFORMER.episode_horizon:
                actions[:, -1] = 0

            outputs = envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            # need to use a deep copy, otherwise, observations would be the same as cleaned_observations
            observations, batch = self._prepare_observation(observations)
            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )

            # reset envs and observations if necessary
            envs_that_needs_to_wait = {}
            for i in range(envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)
                # This helps us to generate the transformer sequence
                # episodes[i].append((cleaned_observations[i], prev_actions[i, -1].item()))
                if not dones[i]:
                    envs_that_needs_to_wait[i] = False
                    continue
                envs_that_needs_to_wait[i] = True
                episodes[i] = []
                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                # observations[i] = envs.reset_at(i)[0]
                # This step is usually done in self._prepare_observation(observations)
                # but now, because we amenbd only one observation, we need to take care of this step manually...
                # observations[i][self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID] = \
                # observations[i][self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID]["tokens"]
                # self.rgb_features = self.rgb_features.set_(torch.zeros((1,), device="cpu"))
                # self.depth_features = self.depth_features.set_(torch.zeros((1,), device="cpu"))
                # observations, batch = self._prepare_observation(observations)

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)
                if has_env_finished_early(envs_that_needs_to_wait):
                    if envs_that_needs_to_wait[i] and i not in envs_to_pause:
                        envs_to_pause.append(i)

            episodes = self._filter_envs_episodes(envs_to_pause, envs, episodes)
            (
                envs,
                hidden_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                hidden_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
            )

            # at this stage, if we dont have any env left,
            # that means that all prediction within the same "batch"
            # are finished, we can wake all envs now.
            if envs.num_envs < 1:
                envs.resume_all()
                episodes = [[] for _ in range(envs.num_envs)]
                rgb_frames = [[] for _ in range(envs.num_envs)]
                observations = envs.reset()
                prev_actions = None
                observations, batch = self._prepare_observation(observations)

        envs.close()
        gc.collect()
        self._release_hook()
        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (
                sum(v[k] for v in stats_episodes.values()) / num_episodes
            )

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        checkpoint_num = checkpoint_index + 1
        for k, v in aggregated_stats.items():
            logger.info(f"{k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)

    def train(self) -> None:
        """Main method for training DAgger."""
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True)
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        else:
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.config.IL.DAGGER.lmdb_map_size),
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        EPS = self.config.IL.DAGGER.expert_policy_sensor
        if EPS not in self.config.TASK_CONFIG.TASK.SENSORS:
            self.config.TASK_CONFIG.TASK.SENSORS.append(EPS)

        self.config.defrost()

        # if doing teacher forcing, don't switch the scene until it is complete
        if self.config.IL.DAGGER.p == 1.0:
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
                -1
            )
        self.config.freeze()

        def collate_fn(batch):
            """Each sample in batch: (
                obs,
                prev_actions,
                oracle_actions,
                inflec_weight,
            )
            """

            def _pad_helper(t, max_len, fill_val=0):
                pad_amount = max_len - t.size(0)
                if pad_amount == 0:
                    return t

                pad = torch.full_like(t[0:1], fill_val).expand(
                    pad_amount, *t.size()[1:]
                )
                return torch.cat([t, pad], dim=0)

            if not _is_correct_previous_actions(batch) and not self.config.IL.DECISION_TRANSFORMER.use_oracle_actions:
                raise Exception(
                    "Dataset has not been created correctly! Prev actions and corrected actions not shifted accordingly!")
            transposed = list(zip(*batch))
            observations_batch = list(transposed[0])

            if self.config.IL.DECISION_TRANSFORMER.recompute_reward:
                for o in observations_batch:
                    scaling_factor = len(o["raw_sparse_reward"])
                    self._calculate_rewards(o, scaling_factor)

            prev_actions_batch = list(transposed[1])
            corrected_actions_batch = list(transposed[2])
            weights_batch = list(transposed[3])  # to make it batch * seq length
            batch_size = len(prev_actions_batch)

            new_observations_batch = defaultdict(list)
            for sensor in observations_batch[0]:
                for bid in range(batch_size):
                    new_observations_batch[sensor].append(
                        observations_batch[bid][sensor]
                    )

            observations_batch = new_observations_batch

            max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
            for bid in range(batch_size):
                for sensor in observations_batch:
                    fill = 0.0 if "_reward" in sensor else 1.0
                    # Workaround when the reward is only a single scalar...
                    if len(observations_batch[sensor][bid].shape) == 0:
                        observations_batch[sensor][bid] = observations_batch[sensor][bid].unsqueeze(-1)
                    observations_batch[sensor][bid] = _pad_helper(
                        observations_batch[sensor][bid], max_traj_len, fill_val=fill
                    )

                prev_actions_batch[bid] = _pad_helper(
                    prev_actions_batch[bid], max_traj_len
                )
                corrected_actions_batch[bid] = _pad_helper(
                    corrected_actions_batch[bid], max_traj_len
                )
                weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

            stack_dimension = 0

            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=stack_dimension)
                if "_reward" in sensor:
                    observations_batch[sensor] = observations_batch[sensor].unsqueeze(-1)

            prev_actions_batch = torch.stack(prev_actions_batch, dim=stack_dimension)
            corrected_actions_batch = torch.stack(corrected_actions_batch, dim=stack_dimension)

            weights_batch = torch.stack(weights_batch, dim=stack_dimension)
            not_done_masks = torch.ones_like(
                corrected_actions_batch, dtype=torch.uint8
            )
            not_done_masks[0] = 0

            if self.config.MODEL.DECISION_TRANSFORMER.use_extra_start_token:
                # The environment only use actions from 0 to 3, the 4 is just a
                # a dummy token to indicate the beginning of a sequence.
                prev_actions_batch[:, 0] = EXTRA_START_TOKEN_ID
            else:
                prev_actions_batch[:, 0] = STOP_ACTION_TOKEN_ID  # this is zero.
            # shape batch size time max episode length
            timesteps = torch.arange(0, max_traj_len).repeat(batch_size, 1)
            observations_batch["timesteps"] = timesteps
            observations_batch = ObservationsDict(observations_batch)

            return (
                observations_batch,
                prev_actions_batch,
                not_done_masks,
                corrected_actions_batch,
                weights_batch
            )

        observation_space, action_space = self._get_spaces(self.config)

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )
        # Seems to bottleneck on Dataloader access if I have more than 1 worker
        workers = self.config.IL.dataload_workers

        # Tries to name the next checkpoints correctly based on the loaded file
        start_epoch = 0
        if self.config.IL.load_from_ckpt and self.config.IL.continue_ckpt_naming:
            checkpoint_name = self.config.IL.ckpt_to_load.split("/")[-1]
            epochs = re.findall(r"\d+", checkpoint_name)
            if len(epochs) > 0:
                start_epoch = int(epochs[0]) + 1

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            for dagger_it in range(self.config.IL.DAGGER.iterations):
                step_id = 0
                if not self.config.IL.DAGGER.preload_lmdb_features:
                    update_id = dagger_it + (1 if self.config.IL.load_from_ckpt else 0)
                    self._update_dataset(
                        update_id
                    )

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()

                dataset = IWTrajectoryDataset(
                    self.lmdb_features_dir,
                    self.config.IL.use_iw,
                    inflection_weight_coef=self.config.IL.inflection_weight_coef,
                    lmdb_map_size=self.config.IL.DAGGER.lmdb_map_size,
                    batch_size=self.config.IL.batch_size,
                    preload_size=self.config.IL.preload_dataloader_size,
                )
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.IL.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=workers,
                )
                num_batch = dataset.length // dataset.batch_size
                print("DAGGER Iteration", dagger_it, "dataset length: ", dataset.length)
                if num_batch == 0:
                    num_batch = 1
                logger.info(f"Number of batches to process:{num_batch}")
                # AuxLosses.activate()
                for epoch in tqdm.trange(
                    self.config.IL.epochs, dynamic_ncols=True
                ):
                    total_loss = 0.0
                    for batch in tqdm.tqdm(
                        diter,
                        total=num_batch,
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        epoch = start_epoch + epoch
                        (
                            observations_batch,
                            prev_actions_batch,
                            not_done_masks,
                            corrected_actions_batch,
                            weights_batch,
                        ) = batch

                        observations_batch = {
                            k: (v.to(
                                device=self.device,
                                non_blocking=True,
                            ) if v.dtype == torch.long else v.to(device=self.device, dtype=torch.float32,
                                                                 non_blocking=True))
                            for k, v in observations_batch.items()
                        }

                        loss, action_loss, aux_loss = self._update_agent(
                            observations_batch,
                            prev_actions_batch.to(
                                device=self.device, non_blocking=True
                            ),
                            not_done_masks.to(
                                device=self.device, non_blocking=True
                            ),
                            corrected_actions_batch.to(
                                device=self.device, non_blocking=True
                            ),
                            weights_batch.to(
                                device=self.device, non_blocking=True
                            ),
                        )

                        writer.add_scalar(
                            f"train_loss_iter_{dagger_it}", loss, step_id
                        )
                        writer.add_scalar(
                            f"train_action_loss_iter_{dagger_it}",
                            action_loss,
                            step_id,
                        )
                        total_loss += loss
                        step_id += 1  # noqa: SIM113
                    total_loss = total_loss / (num_batch)
                    writer.add_scalar(
                        f"train_total_loss{dagger_it}",
                        total_loss,
                        epoch,
                    )
                    logger.info(f"Mean Loss for DAgger iter {dagger_it}, Epoch {epoch}: {total_loss}")
                    if total_loss <= self.config.IL.mean_loss_to_save_checkpoint and (
                        epoch + 1) % self.config.IL.checkpoint_frequency == 0:
                        print("Save", f"ckpt.{dagger_it * self.config.IL.epochs + epoch}.pth")
                        self.save_checkpoint(
                            f"ckpt.{dagger_it * self.config.IL.epochs + epoch}.pth"
                        )
                    if total_loss <= self.config.IL.mean_loss_to_stop_training:
                        logger.info(f"Stopping training early at epoch {epoch}")
                        break
                # AuxLosses.deactivate()



    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        """
        Returns the agent loss in the training loop
        :param observations:
        :param prev_actions:
        :param not_done_masks:
        :param corrected_actions:
        :param weights:
        :param step_grad:
        :param loss_accumulation_scalar:
        :return:
        """

        hidden_states = None

        distribution = self.policy.build_distribution(
            observations, hidden_states, prev_actions, not_done_masks
        )

        logits = distribution.logits

        # The permutation allows to keep the expected input shape (batch times classes)
        # the third dimension gets interpreted as a sequence correctly, as the target actions
        # have the correct shape
        action_loss = F.cross_entropy(
            logits.permute(0, 2, 1), corrected_actions, reduction="none"
        )
        action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        aux_loss = 0.0
        loss = action_loss
        loss = loss / loss_accumulation_scalar
        loss.backward()

        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.item()
        return loss.item(), action_loss.item(), aux_loss
