# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An extension of Rainbow to perform quantile regression.

This loss is computed as in "Distributional Reinforcement Learning with Quantile
Regression" - Dabne et. al, 2017"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.metrics import statistics_instance
from dopamine.replay_memory import prioritized_replay_buffer
import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from ensemble_rl.vectorized_buffer import VectorizedOutOfGraphReplayBuffer
from ensemble_rl.networks import QuantileNetworkEnsemble
import flax
from pathlib import Path
import pickle
import wandb
from typing import Optional

@functools.partial(
    jax.jit,
    static_argnames=('network_def', 'num_actions', 'eval_mode', 'epsilon_eval',
                     'epsilon_train', 'epsilon_decay_period',
                     'min_replay_history', 'epsilon_fn', 'agent_id',
                     'network_def_single'))
def select_action(network_def, params, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn, agent_id,
                  network_def_single):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.
  
    Args:
      network_def: Linen Module to use for inference.
      params: Linen params (frozen dict) to use for inference.
      state: input state to use for inference.
      rng: Jax random number generator.
      num_actions: int, number of actions (static_argnum).
      eval_mode: bool, whether we are in eval mode (static_argnum).
      epsilon_eval: float, epsilon value to use in eval mode (static_argnum).
      epsilon_train: float, epsilon value to use in train mode (static_argnum).
      epsilon_decay_period: float, decay period for epsilon value for certain
        epsilon functions, such as linearly_decaying_epsilon, (static_argnum).
      training_steps: int, number of training steps so far.
      min_replay_history: int, minimum number of steps in replay buffer
        (static_argnum).
      epsilon_fn: function used to calculate epsilon value (static_argnum).
  
    Returns:
      rng: Jax random number generator.
      action: int, the selected action.
    """
    epsilon = jnp.where(
        eval_mode, epsilon_eval,
        epsilon_fn(epsilon_decay_period, training_steps, min_replay_history,
                   epsilon_train))

    rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
    p = jax.random.uniform(rng1)

    if eval_mode:
        # We measure entropy no matter what
        # (E, A) or (E, E, A)
        q_values = network_def.apply(params, state).q_values

        # Compute entropy
        # (E)
        member_action = jnp.argmax(q_values, axis=-1)
        # One hot (E, A)
        member_action_one_hot = jax.nn.one_hot(member_action,
                                               num_classes=num_actions,
                                               axis=-1)
        # (E, A) -> (A)
        votes = member_action_one_hot.sum(axis=0)
        assert votes.shape == (num_actions, )
        if agent_id is None:
            # Voting, break ties randomly
            max_mask = (votes == votes.max(axis=0))
            # Normalized
            max_mask_normalized = max_mask / max_mask.sum(axis=0)
            assert votes.ndim == 1
            all_actions = jnp.arange(votes.shape[0])
            action = jax.random.choice(key=rng3, a=all_actions, p=max_mask_normalized)
            # action = jnp.argmax(votes, axis=0)
            q_value = None
        else:
            # Get one agent for acting
            action = jnp.argmax(q_values[agent_id], axis=0)
            q_value = jnp.max(q_values[agent_id], axis=0)  # for loggin

        # Entropy
        normalized = votes / network_def.ensemble_size
        # Handles 0*log 0
        entropy = -jnp.sum(
            normalized * jnp.log(normalized), axis=0, where=(votes != 0.0))
    else:
        # Training mode, only evaluate one network. Code is extremely confusing but this is faster
        assert agent_id is not None
        # (E, A) -> ()
        params = flax.core.unfreeze(params)
        # Extract encoder, if necessary


        params['params'] = jax.tree_map(lambda x: x[agent_id][None], params['params'])
        params = flax.core.freeze(params)

        q_values = network_def_single.apply(params, state).q_values
        # (1, A) -> (A)
        q_values = q_values.squeeze(axis=0)
        action = jnp.argmax(q_values, axis=0)
        entropy = None
        q_value = jnp.max(q_values, axis=0)  # for loggin

    info = {'q_value': q_value, 'entropy': entropy}

    return rng, jnp.where(p <= epsilon,
                          jax.random.randint(rng2, (), 0, num_actions),
                          action), info

@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
def target_distribution(target_network, next_states, rewards, terminals,
                        cumulative_gamma):
    """Builds the Quantile target distribution as per Dabney et al. (2017).

  Args:
    target_network: Jax Module used for the target network.
    next_states: numpy array of batched next states.
    rewards: numpy array of batched rewards.
    terminals: numpy array of batched terminals.
    cumulative_gamma: float, cumulative gamma to use.

  Returns:
    The target distribution from the replay.
  """
    is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
    # Incorporate terminal state to discount factor.
    gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
    next_state_target_outputs = target_network(next_states)
    q_values = (next_state_target_outputs.q_values)
    assert q_values.shape == next_state_target_outputs.q_values.shape
    # (E)
    next_qt_argmax = jnp.argmax(q_values, axis=-1)
    # (E, A, Q)
    logits = (next_state_target_outputs.logits)
    assert logits.shape == next_state_target_outputs.logits.shape
    # (E, Q)
    next_logits = jax.vmap(lambda l, a: l[a])(logits, next_qt_argmax)
    return jax.lax.stop_gradient(rewards + gamma_with_terminal * next_logits)


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12))
def train(network_def, online_params, target_params, optimizer,
          optimizer_state, states, actions, next_states, rewards, terminals,
          kappa, num_atoms, cumulative_gamma):
    """Run a training step."""
    def loss_fn(params, target):
        def q_online(state):
            return network_def.apply(params, state)

        # (B, E, A, Q)
        logits = jax.vmap(q_online)(states).logits
        logits = (logits)
        # Fetch the logits for its selected action. We use vmap to perform this
        # indexing across the batch.
        # (B, E, Q)
        chosen_action_logits = jax.vmap(lambda x, y: x[:, y, :])(logits, actions)
        # (B, E, Q, Q)
        bellman_errors = (target[:, :, None, :] - chosen_action_logits[:, :, :, None]
                          )  # Input `u' of Eq. 9.
        # Eq. 9 of paper.
        huber_loss = (
            (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) * 0.5 *
            bellman_errors**2 +
            (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) * kappa *
            (jnp.abs(bellman_errors) - 0.5 * kappa))

        tau_hat = ((jnp.arange(num_atoms, dtype=jnp.float32) + 0.5) / num_atoms
                   )  # Quantile midpoints.  See Lemma 2 of paper.
        # Eq. 10 of paper.
        tau_bellman_diff = jnp.abs(tau_hat[None, None, :, None] -
                                   (bellman_errors < 0).astype(jnp.float32))
        quantile_huber_loss = tau_bellman_diff * huber_loss
        # Sum over tau dimension, average over target value dimension.
        loss = jnp.sum(jnp.mean(quantile_huber_loss, axis=-1), axis=(-1, -2))
        assert loss.ndim == 1
        return jnp.mean(loss), loss

    def q_target(state):
        return network_def.apply(target_params, state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    target = target_distribution(q_target, next_states, rewards, terminals,
                                 cumulative_gamma)
    (mean_loss, loss), grad = grad_fn(online_params, target)
    updates, optimizer_state = optimizer.update(grad,
                                                optimizer_state,
                                                params=online_params)
    online_params = optax.apply_updates(online_params, updates)
    info = {}
    return optimizer_state, online_params, loss, mean_loss, info


@gin.configurable
class JaxBootQuantileAgent(dqn_agent.JaxDQNAgent):
    """An implementation of Quantile regression DQN agent."""
    def __init__(self,
                 num_actions,
                 ensemble_size,
                 tandem: bool,
                 active_prob: Optional[None],
                 observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
                 stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
                 network=QuantileNetworkEnsemble,
                 kappa=1.0,
                 num_atoms=200,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=50000,
                 update_period=4,
                 target_update_period=10000,
                 epsilon_fn=dqn_agent.linearly_decaying_epsilon,
                 epsilon_train=0.1,
                 epsilon_eval=0.05,
                 epsilon_decay_period=1000000,
                 replay_scheme='prioritized',
                 optimizer='adam',
                 summary_writer=None,
                 summary_writing_frequency=500,
                 seed=None,
                 allow_partial_reload=False):
        """Initializes the agent and constructs the Graph.

    Args:
      num_actions: Int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.linen Module, expects 3 parameters: num_actions, num_atoms,
        network_type.
      kappa: Float, Huber loss cutoff.
      num_atoms: Int, the number of buckets for the value function distribution.
      gamma: Float, exponential decay factor as commonly used in the RL
        literature.
      update_horizon: Int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: Int, number of stored transitions for training to
        start.
      update_period: Int, period between DQN updates.
      target_update_period: Int, ppdate period for the target network.
      epsilon_fn: Function expecting 4 parameters: (decay_period, step,
        warmup_steps, epsilon), and which returns the epsilon value used for
        exploration during training.
      epsilon_train: Float, final epsilon for training.
      epsilon_eval: Float, epsilon during evaluation.
      epsilon_decay_period: Int, number of steps for epsilon to decay.
      replay_scheme: String, replay memory scheme to be used. Choices are:
        uniform - Standard (DQN) replay buffer (Mnih et al., 2015)
        prioritized - Prioritized replay buffer (Schaul et al., 2015)
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      seed: int, a seed for DQN's internal RNG, used for initialization and
        sampling actions. If None, will use the current time in nanoseconds.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
        self._num_atoms = num_atoms
        self._kappa = kappa
        self._replay_scheme = replay_scheme
        self.ensemble_size = ensemble_size
        self.tandem = tandem
        self.active_prob = active_prob

        super(JaxBootQuantileAgent, self).__init__(
            num_actions=num_actions,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            stack_size=stack_size,
            network=functools.partial(network, num_atoms=num_atoms, ensemble_size=self.ensemble_size),
            gamma=gamma,
            update_horizon=update_horizon,
            min_replay_history=min_replay_history,
            update_period=update_period,
            target_update_period=target_update_period,
            epsilon_fn=epsilon_fn,
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            optimizer=optimizer,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency,
            seed=seed,
            allow_partial_reload=allow_partial_reload)

        self.network_def_single = network(
            num_actions=self.num_actions,
            num_atoms=num_atoms,
            inputs_preprocessed=self.network_def.inputs_preprocessed,
            ensemble_size=1
        )
        self.update_info = {}

    def _build_networks_and_optimizer(self):
        self._rng, rng = jax.random.split(self._rng)
        self.online_params = self.network_def.init(rng, x=self.state)
        self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = self.online_params

    def _build_replay_buffer(self):
        """Creates the replay buffer used by the agent."""
        # if self._replay_scheme not in ['uniform', 'prioritized']:
            # raise ValueError('Invalid replay scheme: {}'.format(
                # self._replay_scheme))
        # Both replay schemes use the same data structure, but the 'uniform' scheme
        # sets all priorities to the same value (which yields uniform sampling).
        # return prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
        #     observation_shape=self.observation_shape,
        #     stack_size=self.stack_size,
        #     update_horizon=self.update_horizon,
        #     gamma=self.gamma,
        #     observation_dtype=self.observation_dtype)
        assert self._replay_scheme == 'uniform'
        return VectorizedOutOfGraphReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype,
            # extra_storage_types=extra_storage_types
        )

    def _train_step(self):
        """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_params to target_network_params if training
    steps is a multiple of target update period.
    """
        if self._replay.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                self._sample_from_replay_buffer()
                self.optimizer_state, self.online_params, loss, mean_loss, info = train(
                    self.network_def, self.online_params,
                    self.target_network_params, self.optimizer,
                    self.optimizer_state,
                    self.preprocess_fn(self.replay_elements['state']),
                    self.replay_elements['action'],
                    self.preprocess_fn(self.replay_elements['next_state']),
                    self.replay_elements['reward'],
                    self.replay_elements['terminal'], self._kappa,
                    self._num_atoms, self.cumulative_gamma)
                if self._replay_scheme == 'prioritized':
                    # The original prioritized experience replay uses a linear exponent
                    # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
                    # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
                    # suggested a fixed exponent actually performs better, except on Pong.
                    probs = self.replay_elements['sampling_probabilities']
                    loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
                    loss_weights /= jnp.max(loss_weights)

                    # Rainbow and prioritized replay are parametrized by an exponent
                    # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
                    # leave it as is here, using the more direct sqrt(). Taking the square
                    # root "makes sense", as we are dealing with a squared loss.  Add a
                    # small nonzero value to the loss to avoid 0 priority items. While
                    # technically this may be okay, setting all items to 0 priority will
                    # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
                    self._replay.set_priority(self.replay_elements['indices'],
                                              jnp.sqrt(loss + 1e-10))

                    # Weight the loss by the inverse priorities.
                    loss = loss_weights * loss
                    mean_loss = jnp.mean(loss)
                info.update({'DQNLoss': mean_loss})
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

    def set_acting_agent_id(self):
        if self.eval_mode:
            if self.eval_policy == 'random':
                self.eval_agent_id = np.random.choice(self.ensemble_size)
        else:
            if self.tandem:
                assert self.active_prob is not None
                assert self.ensemble_size == 2
                self.current_agent_id = np.random.choice(
                    self.ensemble_size,
                    p=[self.active_prob, 1.0 - self.active_prob])
            else:
                self.current_agent_id = np.random.choice(
                    self.ensemble_size)

    def get_acting_agent_id(self):
        if self.eval_mode:
            if self.eval_policy == 'single':
                assert self.eval_agent_id is not None
                agent_id = self.eval_agent_id  # will be set externally
            elif self.eval_policy == 'random':
                assert self.eval_agent_id is not None  # will be reset at the start of episode
                agent_id = self.eval_agent_id
            else:
                assert self.eval_policy == 'vote'
                # Voting
                agent_id = None
        else:
            agent_id = self.current_agent_id
        return agent_id

    def reset_entropy_list(self):
        self.entropy_list = []

    def set_eval_policy(self,
                        eval_policy: str,
                        eval_agent_id: Optional[int] = None):
        assert eval_policy in ['vote', 'random', 'single']
        self.eval_policy = eval_policy
        self.eval_agent_id = eval_agent_id
        if eval_policy == 'single':
            assert eval_agent_id is not None
        else:
            assert eval_agent_id is None

    def begin_episode(self, observation):
        """Returns the agent's first action for this episode.
  
      Args:
        observation: numpy array, the environment's initial observation.
  
      Returns:
        int, the selected action.
      """
        self._reset_state()
        self._record_observation(observation)

        if not self.eval_mode:
            self._train_step()

        # Sample random agent (if necessary)
        self.set_acting_agent_id()
        agent_id = self.get_acting_agent_id()
        self._rng, self.action, info = select_action(
            self.network_def,
            self.online_params,
            self.preprocess_fn(self.state),
            self._rng,
            self.num_actions,
            self.eval_mode,
            self.epsilon_eval,
            self.epsilon_train,
            self.epsilon_decay_period,
            self.training_steps,
            self.min_replay_history,
            self.epsilon_fn,
            agent_id=agent_id,
            network_def_single=self.network_def_single,
        )
        if self.eval_mode:
            entropy = info.pop('entropy')
            assert entropy is not None
            entropy = entropy.item()
            self.entropy_list.append(entropy)
        self.action = np.asarray(self.action)
        return self.action, info

    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.
  
      We store the observation of the last time step since we want to store it
      with the reward.
  
      Args:
        reward: float, the reward received from the agent's most recent action.
        observation: numpy array, the most recent observation.
  
      Returns:
        int, the selected action.
      """
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            self._store_transition(self._last_observation, self.action, reward,
                                   False)
            self._train_step()

        agent_id = self.get_acting_agent_id()
        self._rng, self.action, info = select_action(
            self.network_def,
            self.online_params,
            self.preprocess_fn(self.state),
            self._rng,
            self.num_actions,
            self.eval_mode,
            self.epsilon_eval,
            self.epsilon_train,
            self.epsilon_decay_period,
            self.training_steps,
            self.min_replay_history,
            self.epsilon_fn,
            agent_id=agent_id,
            network_def_single=self.network_def_single,
        )

        if self.eval_mode:
            entropy = info.pop('entropy')
            assert entropy is not None
            entropy = entropy.item()
            self.entropy_list.append(entropy)
        self.action = np.asarray(self.action)
        return self.action, info

    def save(self, checkpoint_dir, filename):
        """
        minimal save
      """
        # if not tf.io.gfile.exists(checkpoint_dir):
            # return None
        # Checkpoint the out-of-graph replay buffer.
        bundle_dictionary = {
            'training_steps': self.training_steps,
            'online_params': self.online_params,
        }
        with (Path(checkpoint_dir) / filename).open('wb') as f:
            pickle.dump(bundle_dictionary, f)

    def load(self, checkpoint_dir, filename):
        with (Path(checkpoint_dir) / filename).open('rb') as f:
            bundle_dictionary = pickle.load(f)
            self.online_params = bundle_dictionary.pop('online_params')
        return bundle_dictionary

