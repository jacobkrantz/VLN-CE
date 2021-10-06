import contextlib
import json
import os
import random
import time
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from gym import Space, spaces
from habitat import Config, logger
from habitat.core.vector_env import VectorEnv
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import batch_obs, linear_decay
from torch import Tensor
from torch import distributed as distrib
from torch import nn as nn
from torch.optim.lr_scheduler import LambdaLR

from habitat_extensions.utils import (
    generate_video,
    waypoint_observations_to_image,
)
from vlnce_baselines.common.ddppo_alg import WDDPPO
from vlnce_baselines.common.env_utils import (
    construct_envs,
    construct_envs_auto_reset_false,
)
from vlnce_baselines.common.rollout_storage import ActionDictRolloutStorage
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.config.default import add_pano_sensors_to_config


@baseline_registry.register_trainer(name="ddppo-waypoint")
class DDPPOWaypointTrainer(PPOTrainer):
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config: Optional[Config] = None) -> None:
        if config.ENV_NAME in [
            "VLNCEWaypointEnv",
            "VLNCEWaypointEnvDiscretized",
        ]:
            config = add_pano_sensors_to_config(config)

        self.video_in_env = config.ENV_NAME == "VLNCEWaypointEnvDiscretized"

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            config = interrupted_state["config"]

        super().__init__(config)

    def _set_observation_space(self, envs, batch, instruction_uuid):
        obs_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], self.obs_transforms
        )

        obs_space.spaces[instruction_uuid] = spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=tuple(batch[instruction_uuid].shape[1:]),
            dtype=np.float,
        )
        obs_space = spaces.Dict(
            {
                **{
                    "rgb_history": obs_space.spaces["rgb"],
                    "depth_history": obs_space.spaces["depth"],
                },
                **obs_space.spaces,
            }
        )
        for k in ["rgb_history", "depth_history"]:
            obs_space.spaces[k] = spaces.Box(
                low=obs_space.spaces[k].low.item(0),
                high=obs_space.spaces[k].high.item(0),
                shape=obs_space.spaces[k].shape[1:],
                dtype=obs_space.spaces[k],
            )
        self.obs_space = obs_space

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        ckpt_to_load: str = None,
    ) -> None:
        policy = baseline_registry.get_policy(config.MODEL.policy_name)
        policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        policy.to(self.device)

        if config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(policy.critic.fc.weight)
            nn.init.constant_(policy.critic.fc.bias, 0)

        ppo_cfg = config.RL.PPO
        self.agent = WDDPPO(
            actor_critic=policy,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            use_clipped_value_loss=ppo_cfg.clip_value_loss,
            offset_regularize_coef=ppo_cfg.offset_regularize_coef,
            pano_entropy_coef=ppo_cfg.pano_entropy_coef,
            offset_entropy_coef=ppo_cfg.offset_entropy_coef,
            distance_entropy_coef=ppo_cfg.distance_entropy_coef,
        )

        if load_from_ckpt:
            ckpt_dict = self.load_checkpoint(ckpt_to_load, map_location="cpu")
            self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.policy = policy

    def _collect_rollout_step(
        self,
        rollouts: ActionDictRolloutStorage,
        current_episode_reward: Tensor,
        running_episode_stats: Dict[str, Tensor],
    ) -> Tuple[
        float, float, int, List[Any], List[bool], DefaultDict[str, List[float]]
    ]:
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            step_prev_actions = {
                k: v[rollouts.step] for k, v in rollouts.prev_actions.items()
            }

            outputs = self.policy.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                step_prev_actions,
                rollouts.masks[rollouts.step],
                deterministic=False,
            )
            (
                values,
                actions,
                action_elements,
                _,
                variances,
                action_log_probs,
                recurrent_hidden_states,
                _,
            ) = outputs

        pth_time += time.time() - t_sample_action
        t_step_env = time.time()

        obs_history = {
            "rgb": torch.zeros_like(step_observation["rgb"][:, 0]),
            "depth": torch.zeros_like(step_observation["depth"][:, 0]),
        }

        logging_predictions = defaultdict(list)
        for i in range(self.envs.num_envs):
            if actions[i]["action"] != "STOP":
                idx = action_elements["pano"][i]
                obs_history["rgb"][i] = step_observation["rgb"][i, idx]
                obs_history["depth"][i] = step_observation["depth"][i, idx]
                logging_predictions["distance_pred"].append(
                    self.policy.net.distance_to_continuous(
                        action_elements["distance"][i]
                    ).item()
                )
                logging_predictions["offset_pred"].append(
                    self.policy.net.offset_to_continuous(
                        action_elements["offset"][i]
                    ).item()
                )
                logging_predictions["distance_var"].append(
                    variances["distance"][i].item()
                    if type(variances["distance"]) is torch.Tensor
                    else variances["distance"]
                )
                logging_predictions["offset_var"].append(
                    variances["offset"][i].item()
                    if type(variances["offset"]) is torch.Tensor
                    else variances["offset"]
                )

        outputs = self.envs.step(actions)
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()

        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        batch = {k: v.float() for k, v in batch.items()}

        # insert the observation histories
        batch["rgb_history"] = obs_history["rgb"]
        batch["depth_history"] = obs_history["depth"]

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=self.device
        ).unsqueeze(1)

        masks = torch.tensor(
            [[0] if done else [1] for done in dones],
            dtype=torch.uint8,
            device=self.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )
            v = torch.tensor(v, dtype=torch.float, device=self.device)
            running_episode_stats[k] += (1 - masks) * v.unsqueeze(1)

        current_episode_reward *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            action_elements,
            action_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return (
            pth_time,
            env_time,
            self.envs.num_envs,
            dones,
            logging_predictions,
        )

    def _update_agent(
        self, ppo_cfg: Config, rollouts: ActionDictRolloutStorage
    ) -> Tuple[float, float, float, float, float, float, float]:
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            prev_actions = {
                k: v[rollouts.step] for k, v in rollouts.prev_actions.items()
            }
            next_value = self.policy.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                prev_actions,
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )
        update_results = self.agent.update(rollouts)
        rollouts.after_update()
        return (time.time() - t_update_model, *update_results)

    def train(self) -> None:
        """Main method for training DD-PPO."""
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )
        add_signal_handlers()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        if self.world_rank == 0:
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_IDS = [self.local_rank]
        # Multiply by NUM_ENVIRONMENTS to ensure simulators get unique seeds
        self.config.TASK_CONFIG.SEED += (
            self.world_rank * self.config.NUM_ENVIRONMENTS
        )
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            workers_ignore_signals=True,
        )
        observations = self.envs.reset()
        instruction_uuid = self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        observations = extract_instruction_tokens(
            observations, instruction_uuid
        )
        batch = batch_obs(observations, device=self.device)
        self.obs_transforms = get_active_obs_transforms(self.config)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        batch["rgb_history"] = batch["rgb"][:, 0].detach().clone() * 0.0
        batch["depth_history"] = batch["depth"][:, 0].detach().clone() * 0.0
        self._set_observation_space(self.envs, batch, instruction_uuid)

        self._initialize_policy(
            config=self.config,
            load_from_ckpt=False,
            observation_space=self.obs_space,
            action_space=self.envs.action_spaces[0],
        )

        self.agent.init_distributed(find_unused_params=True)
        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )

        ppo_cfg = self.config.RL.PPO
        rollouts = ActionDictRolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.obs_space,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            num_recurrent_layers=self.policy.net.num_recurrent_layers,
            continuous_offset=self.config.MODEL.WAYPOINT.continuous_offset,
            continuous_distance=self.config.MODEL.WAYPOINT.continuous_distance,
        )
        rollouts.to(self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )
        window_logging_predictions = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.RL.NUM_UPDATES),
        )

        filename = None
        if self.config.RL.DDPPO.start_from_requeue:
            filename = self.config.RL.DDPPO.requeue_path
            # Set to False so future requeues don't load old values
            self.config.defrost()
            self.config.RL.DDPPO.start_from_requeue = False
            self.config.freeze()

        interrupted_state = load_interrupted_state(filename)
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(start_update, self.config.RL.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.RL.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            )
                        )

                    requeue_job()
                    return

                count_steps_delta = 0
                self.agent.eval()
                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                        dones,
                        logging_predictions,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    for k, v in logging_predictions.items():
                        window_logging_predictions[k].extend(v)

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step
                        >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break

                num_rollouts_done_store.add("num_done", 1)

                self.agent.train()

                # batchnorm layers continue to update otherwise
                self.policy.net.rgb_encoder.eval()
                self.policy.net.depth_encoder.eval()

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    entropy_loss,
                    pano_entropy,
                    offset_entropy,
                    distance_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)

                pth_time += delta_pth_time
                stats_ordering = sorted(running_episode_stats.keys())
                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )
                distrib.all_reduce(stats)

                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                    [
                        value_loss,
                        action_loss,
                        entropy_loss,
                        pano_entropy,
                        offset_entropy,
                        distance_entropy,
                        count_steps_delta,
                    ],
                    device=self.device,
                )
                distrib.all_reduce(stats)

                # only rank 0 does logging and checkpointing
                if self.world_rank != 0:
                    continue

                count_steps += stats[6].item()
                stats /= self.world_size

                num_rollouts_done_store.set("num_done", "0")
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                reward = deltas["reward"] / deltas["count"]
                writer.add_scalar("reward", reward, count_steps)

                ignore = {"reward", "count", "waypoint_reward_measure"}
                if len(deltas) > len(ignore):
                    metrics = {
                        k: v / deltas["count"]
                        for k, v in deltas.items()
                        if k not in ignore
                    }
                    writer.add_scalars("metrics", metrics, count_steps)

                if len(window_logging_predictions):
                    preds = {
                        k: np.mean(v)
                        for k, v in window_logging_predictions.items()
                        if len(v)
                    }
                    writer.add_scalars("predictions", preds, count_steps)

                losses = {
                    "value": stats[0].item(),
                    "policy": stats[1].item(),
                    "entropy": stats[2].item(),
                }
                writer.add_scalars("losses", losses, count_steps)

                entropies = {
                    "pano_entropy": stats[3].item(),
                    "offset_entropy": stats[4].item(),
                    "distance_entropy": stats[5].item(),
                }
                if not self.config.MODEL.WAYPOINT.predict_offset:
                    del entropies["offset_entropy"]
                if not self.config.MODEL.WAYPOINT.predict_distance:
                    del entropies["distance_entropy"]
                writer.add_scalars("entropies", entropies, count_steps)

                if update > 0 and update % self.config.RL.LOG_INTERVAL == 0:
                    fps = count_steps / ((time.time() - t_start) + prev_time)
                    logger.info(
                        f"update: {update}"
                        f"\tfps: {fps:.2f}"
                        f"\tenv-time: {env_time:.2f}s"
                        f"\tpth-time: {pth_time:.2f}s"
                        f"\tframes: {int(count_steps)}"
                    )
                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                if update % self.config.RL.CHECKPOINT_INTERVAL == 0:
                    requeue_stats = dict(
                        env_time=env_time,
                        pth_time=pth_time,
                        count_steps=count_steps,
                        count_checkpoints=count_checkpoints,
                        start_update=update,
                        prev_time=(time.time() - t_start) + prev_time,
                    )
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=count_steps,
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            requeue_stats=requeue_stats,
                        ),
                    )
                    count_checkpoints += 1

            self.envs.close()

    @staticmethod
    def _pause_envs(
        envs_to_pause: List[Any],
        envs: VectorEnv,
        rnn_states: Tensor,
        prev_actions: Dict[str, Tensor],
        not_done_masks: Tensor,
        batch: Dict[str, Tensor],
        obs_history: Dict[str, Tensor],
    ) -> Tuple[
        VectorEnv,
        Tensor,
        Dict[str, Tensor],
        Tensor,
        Dict[str, Tensor],
        Dict[str, Tensor],
    ]:
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            rnn_states = rnn_states[state_index]
            not_done_masks = not_done_masks[state_index]
            if isinstance(prev_actions, dict):
                for k in prev_actions:
                    prev_actions[k] = prev_actions[k][state_index]
            else:
                prev_actions = prev_actions[state_index]
            for k in obs_history:
                obs_history[k] = obs_history[k][state_index]
            for k, v in batch.items():
                batch[k] = v[state_index]

        return (
            envs,
            rnn_states,
            prev_actions,
            not_done_masks,
            batch,
            obs_history,
        )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """Evaluates a single checkpoint of a waypoint-based model.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")

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

        if config.VIDEO_OPTION:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.SENSORS.append("GLOBAL_GPS_SENSOR")
            config.TASK_CONFIG.TASK.SENSORS.append("HEADING_SENSOR")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            fname = f"stats_ckpt_{checkpoint_index}_{split}.json"
            fname = os.path.join(config.RESULTS_DIR, fname)
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        observations = envs.reset()
        instruction_uuid = config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        instructions = [o[instruction_uuid]["text"] for o in observations]
        observations = extract_instruction_tokens(
            observations, instruction_uuid
        )
        batch = batch_obs(observations, self.device)
        self.obs_transforms = get_active_obs_transforms(self.config)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        batch = {k: v.float() for k, v in batch.items()}
        self._set_observation_space(envs, batch, instruction_uuid)

        self._initialize_policy(
            config=config,
            load_from_ckpt=True,
            observation_space=self.obs_space,
            action_space=envs.action_spaces[0],
            ckpt_to_load=checkpoint_path,
        )
        self.policy.eval()

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = {
            "pano": torch.zeros(
                envs.num_envs,
                1,
                device=self.device,
                dtype=torch.long,
            ),
            "offset": torch.zeros(envs.num_envs, 1, device=self.device),
            "distance": torch.zeros(envs.num_envs, 1, device=self.device),
        }
        obs_history = {
            "rgb": torch.zeros_like(batch["rgb"][:, 0]),
            "depth": torch.zeros_like(batch["depth"][:, 0]),
        }

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        rgb_frames = None
        if config.VIDEO_OPTION:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
            rgb_frames = [[] for _ in range(envs.num_envs)]

        stats_episodes = {}
        infos = [{} for _ in range(envs.num_envs)]

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps)
        while envs.num_envs > 0 and len(stats_episodes) < num_eps:

            batch["rgb_history"] = obs_history["rgb"]
            batch["depth_history"] = obs_history["depth"]

            with torch.no_grad():
                batch_in = {k: v.detach().clone() for k, v in batch.items()}
                deterministic = not self.config.EVAL.SAMPLE
                outputs = self.policy.act(
                    batch_in,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=deterministic,
                )

            (
                _,
                actions,
                action_elements,
                distribution_modes,
                _,
                _,
                rnn_states,
                pano_stop_distribution,
            ) = outputs

            prev_actions = {
                k: v.detach().clone() for k, v in action_elements.items()
            }

            for i in range(envs.num_envs):
                pano_index = prev_actions["pano"][i].item()
                obs_history["rgb"][i] = batch["rgb"][i][
                    pano_index % batch["rgb"][i].size(0)
                ]
                obs_history["depth"][i] = batch["depth"][i][
                    pano_index % batch["depth"][i].size(0)
                ]

            if config.VIDEO_OPTION and not self.video_in_env:
                for i in range(envs.num_envs):
                    vinfo = deepcopy(infos[i])
                    if "top_down_map_vlnce" not in vinfo:
                        vinfo = envs.call_at(
                            i, "get_info", {"observations": {}}
                        )

                    np_obs = {k: v[i].cpu().numpy() for k, v in batch.items()}
                    obs = {"instruction_text": instructions[i], **np_obs}
                    frame = waypoint_observations_to_image(
                        observation=obs,
                        info=vinfo,
                        pano_distribution=pano_stop_distribution.probs[i]
                        .cpu()
                        .numpy(),
                        agent_action_elements={
                            k: v[i].item() for k, v in action_elements.items()
                        },
                        agent_stop=actions[i]["action"] == "STOP",
                        distribution_modes={
                            k: v[i].item()
                            for k, v in distribution_modes.items()
                        },
                        predict_offset=config.MODEL.WAYPOINT.predict_offset,
                        predict_distance=config.MODEL.WAYPOINT.predict_distance,
                        agent_position=batch["globalgps"][i]
                        .cpu()
                        .unsqueeze(0),
                        agent_heading=batch["heading"][i].cpu(),
                        num_panos=config.TASK_CONFIG.TASK.PANO_ROTATIONS,
                    )
                    rgb_frames[i].append(frame)

            current_episodes = envs.current_episodes()

            outputs = envs.step(actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )

            # reset envs, stats, observations if necessary
            for i in range(envs.num_envs):
                if not dones[i]:
                    continue

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]

                pbar.update()

                if config.VIDEO_OPTION and not self.video_in_env:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            "SPL": round(stats_episodes[ep_id]["spl"], 5)
                        },
                        tb_writer=writer,
                        fps=1,
                    )
                    rgb_frames[i] = []

                stats_episodes[ep_id].pop("top_down_map_vlnce", None)

            instructions = [o[instruction_uuid]["text"] for o in observations]
            observations = extract_instruction_tokens(
                observations, instruction_uuid
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            batch = {k: v.float() for k, v in batch.items()}

            envs_to_pause = []
            next_episodes = envs.current_episodes()
            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                envs,
                rnn_states,
                prev_actions,
                not_done_masks,
                batch,
                obs_history,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                rnn_states,
                prev_actions,
                not_done_masks,
                batch,
                obs_history,
            )

        envs.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (
                sum([v[k] for v in stats_episodes.values()]) / num_episodes
            )

        if self.config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        logger.info("Average episode stats:")
        for k, v in aggregated_stats.items():
            logger.info(f"{k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_index)
