from typing import Tuple

import torch
from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO
from torch.functional import Tensor
from torch.nn.functional import l1_loss


class WDDPPO(DDPPO):
    """Differences with DD-PPO:
    - expands entropy calculation and tracking to three variables
    - adds a regularization term to the offset prediction
    """

    def __init__(
        self,
        *args,
        offset_regularize_coef: float = 0.0,
        pano_entropy_coef: float = 1.0,
        offset_entropy_coef: float = 1.0,
        distance_entropy_coef: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.offset_regularize_coef = offset_regularize_coef
        self.pano_entropy_coef = pano_entropy_coef
        self.offset_entropy_coef = offset_entropy_coef
        self.distance_entropy_coef = distance_entropy_coef

    def get_advantages(self, rollouts) -> Tensor:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def update(self, rollouts) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        pano_entropy_epoch = 0.0
        offset_entropy_epoch = 0.0
        distance_entropy_epoch = 0.0

        for _e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                entropy_loss = (
                    self.pano_entropy_coef * entropy["pano"]
                    + self.offset_entropy_coef * entropy["offset"]
                    + self.distance_entropy_coef * entropy["distance"]
                ).mean() * self.entropy_coef

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                value_loss = value_loss * self.value_loss_coef

                # slight regularization to the offset
                offset_loss = 0.0
                if "offset" in actions_batch:
                    offset_loss = self.offset_regularize_coef * l1_loss(
                        self.actor_critic.net.offset_to_continuous(
                            actions_batch["offset"]
                        ),
                        torch.zeros_like(actions_batch["offset"]),
                    )

                self.optimizer.zero_grad()
                loss = value_loss + action_loss + offset_loss - entropy_loss

                self.before_backward(loss)
                loss.backward()
                self.after_backward(loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                pano_entropy_epoch += entropy["pano"].mean().item()
                offset_entropy_epoch += entropy["offset"].mean().item()
                distance_entropy_epoch += entropy["distance"].mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        return (
            value_loss_epoch / num_updates,
            action_loss_epoch / num_updates,
            entropy_loss_epoch / num_updates,
            pano_entropy_epoch / num_updates,
            offset_entropy_epoch / num_updates,
            distance_entropy_epoch / num_updates,
        )
