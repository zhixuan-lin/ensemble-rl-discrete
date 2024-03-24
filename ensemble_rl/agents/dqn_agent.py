from dopamine.jax.agents.dqn.dqn_agent import JaxDQNAgent
from ensemble_rl.vectorized_buffer import VectorizedOutOfGraphReplayBuffer
import wandb
import tensorflow as tf
from dopamine.metrics import statistics_instance
import numpy as onp
import functools
import jax
import jax.numpy as jnp
from dopamine.jax import losses
import optax
from collections import OrderedDict
import gin
import logging


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12))
def train(network_def,
          online_params,
          target_params,
          optimizer,
          optimizer_state,
          states,
          actions,
          next_states,
          rewards,
          terminals,
          cumulative_gamma,
          loss_type,
          double_dqn):
    """Run the training step."""
    def loss_fn(params, target):
        def q_online(state):
            return network_def.apply(params, state)

        q_values = jax.vmap(q_online)(states).q_values
        q_values = jnp.squeeze(q_values)
        replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
        info = {
            # (B,)
            'q_values_data': replay_chosen_q.mean(axis=0),
            # (B, A)
            'q_values_pi': q_values.max(axis=-1).mean(axis=0),
        }
        if loss_type == 'huber':
            return jnp.mean(
                jax.vmap(losses.huber_loss)(target, replay_chosen_q)), info
        return jnp.mean(jax.vmap(losses.mse_loss)(target,
                                                  replay_chosen_q)), info

    def q_online(state):
        return network_def.apply(online_params, state)

    def q_target(state):
        return network_def.apply(target_params, state)

    target = target_q(q_online, q_target, next_states, rewards, terminals,
                      cumulative_gamma, double_dqn)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, info), grad = grad_fn(online_params, target)
    updates, optimizer_state = optimizer.update(grad,
                                                optimizer_state,
                                                params=online_params)
    online_params = optax.apply_updates(online_params, updates)
    return optimizer_state, online_params, loss, info


def target_q(online_network, target_network, next_states, rewards, terminals,
             cumulative_gamma, double_dqn):
    """Compute the target Q-value."""
    if double_dqn:
        next_state_q_vals_for_argmax = jax.vmap(
            online_network, in_axes=(0))(next_states).q_values
    else:
        next_state_q_vals_for_argmax = jax.vmap(
            target_network, in_axes=(0))(next_states).q_values
    # next_state_q_vals_for_argmax = jnp.squeeze(next_state_q_vals_for_argmax)
    # (B, A) -> (B)
    next_argmax = jnp.argmax(next_state_q_vals_for_argmax, axis=-1)
    # (B, A)
    q_values = jax.vmap(target_network, in_axes=(0))(next_states).q_values
    # (B)
    replay_next_qt_max = jax.vmap(lambda t, u: t[..., u])(q_values, next_argmax)
    return jax.lax.stop_gradient(rewards +
                                 cumulative_gamma * replay_next_qt_max *
                                 (1. - terminals))

@gin.configurable
class JaxDQNCustomAgent(JaxDQNAgent):
    def __init__(self, *args, double_dqn, **kwargs):
        super().__init__(*args, **kwargs)
        self.double_dqn = double_dqn
        logging.info('\t double_dqn: %s', self.double_dqn)
        self.update_info = {}

    def _build_replay_buffer(self):
        """Creates the replay buffer used by the agent."""
        return VectorizedOutOfGraphReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype)

    def _train_step(self):
        """Runs a single training step.

        Runs training if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.
  
        Also, syncs weights from online_params to target_network_params if training
        steps is a multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self._replay.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                self._sample_from_replay_buffer()
                states = self.preprocess_fn(self.replay_elements['state'])
                next_states = self.preprocess_fn(
                    self.replay_elements['next_state'])
                self.optimizer_state, self.online_params, loss, info = train(
                    self.network_def, self.online_params,
                    self.target_network_params, self.optimizer,
                    self.optimizer_state, states,
                    self.replay_elements['action'], next_states,
                    self.replay_elements['reward'],
                    self.replay_elements['terminal'], self.cumulative_gamma,
                    self._loss_type, self.double_dqn)

                info.update({'DQNLoss': loss})
                info = {f'Agent/{k}': v for k, v in info.items()}
                self.update_info = info

                should_log = (self.training_steps > 0 and self.training_steps %
                              self.summary_writing_frequency == 0)
                if should_log:
                    info = {k: v.item() for k, v in info.items()}

                    if wandb.run is not None:
                        wandb.log(info, step=self.training_steps)
                    # if self.summary_writer is not None:
                    # with self.summary_writer.as_default():
                    # tf.summary.scalar('Agent/DQNLoss',
                    # loss,
                    # step=self.training_steps)
                    # self.summary_writer.flush()
                    if hasattr(self, 'collector_dispatcher'):
                        self.collector_dispatcher.write(
                            [
                                statistics_instance.StatisticsInstance(
                                    key, value, step=self.training_steps)
                                for key, value in info.items()
                            ],
                            collector_allowlist=self._collector_allowlist)
            if self.training_steps % self.target_update_period == 0:
                self._sync_weights()

        self.training_steps += 1
