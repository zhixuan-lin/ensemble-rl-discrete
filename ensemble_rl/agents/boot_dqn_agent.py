from ensemble_rl.agents.dqn_agent import JaxDQNCustomAgent
import collections
import pickle
import functools
from ensemble_rl.networks import NatureDQNNetworkEnsemble
import jax
import jax.numpy as jnp
from dopamine.jax import losses
import optax
import wandb
from dopamine.metrics import statistics_instance
from typing import Optional
import numpy as onp
import gin
import flax
from flax import core
import logging
import tensorflow as tf
from pathlib import Path
from dopamine.jax.agents.dqn.dqn_agent import create_optimizer
from ensemble_rl.vectorized_buffer import VectorizedOutOfGraphReplayBuffer
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
from ensemble_rl import metric_utils


@functools.partial(jax.jit,
                   static_argnames=('network_def', 'optimizer',
                                    'cumulative_gamma', 'loss_type',
                                    'double_dqn', 'aux_loss', 'grad_scale',
                                    'single_cerl', 'multi_gamma',
                                    'num_gammas', 'cerl_own_target',
                                    'mico', 'mico_weight', 'bootstrap'))
def train(network_def, online_params, target_params, optimizer,
          optimizer_state, states, actions, next_states, rewards, terminals,
          cumulative_gamma, loss_type, double_dqn, aux_loss, grad_scale,
          single_cerl, multi_gamma, num_gammas, cerl_own_target, mico,
          mico_weight, ens_mask, bootstrap):
    """Run the training step."""
    B = states.shape[0]
    A = network_def.num_actions
    E = network_def.ensemble_size

    if actions.ndim > 1:
        assert actions.shape == rewards.shape == terminals.shape == states.shape[:2] == next_states.shape[:2] == (B, E)
        share_batch = False
        assert aux_loss == 'none'
    else:
        assert actions.shape == rewards.shape == terminals.shape == states.shape[:1] == next_states.shape[:1] == (B,)
        assert states.ndim == 4
        share_batch = True

    # When aux_loss != None, shape will be (B, E1, E2, A)
    # E1 indexes the ensemble, E2 indexes each head of the ensemble.
    # Other wise (B, E, A)
    def loss_fn(params, target, target_next_r):
        def q_online(state):
            return network_def.apply(params, state, share_batch=share_batch)

        output = jax.vmap(q_online)(states)
        q_values, representations = output.q_values, output.representation
        if aux_loss == 'none':
            assert q_values.shape == (B, E, A)
        else:
            if not multi_gamma:
                assert q_values.shape == (B, E, E, A)
            else:
                assert q_values.shape == (B, E, num_gammas, A)
        # q_values = jnp.squeeze(q_values)
        if share_batch:
            # (B, E, A) (B,)-> (B, E)
            # or (B, E, E, A) (B,)-> (B, E, E)
            replay_chosen_q = jax.vmap(lambda x, y: x[..., y])(q_values, actions)
        else:
            # (B, E, A) (B, E)-> (B, E)
            replay_chosen_q = jax.vmap(jax.vmap(lambda x, y: x[y]))(q_values, actions)

        if aux_loss == 'none':
            info = {
                # (B, E)
                **{
                    f'q_values_data_{i}': replay_chosen_q[:, i].mean(axis=0)
                    for i in range(E)
                },
                # (B, E, A)
                **{
                    f'q_values_pi_{i}': q_values[:, i, :].max(axis=-1).mean(axis=0)
                    for i in range(E)
                }
            }
        else:
            if not multi_gamma:
                info = {
                    # (B, E)
                    **{
                        f'q_values_data_{i}': replay_chosen_q[:, i, i].mean(axis=0)
                        for i in range(E)
                    },
                    # (B, E, A)
                    **{
                        f'q_values_pi_{i}': q_values[:, i, i, :].max(axis=-1).mean(axis=0)
                        for i in range(E)
                    }
                }
            else:
                info = {
                    # (B, E)
                    **{
                        f'q_values_data_{i}': replay_chosen_q[:, i, 0].mean(axis=0)
                        for i in range(E)
                    },
                    # (B, E, A)
                    **{
                        f'q_values_pi_{i}': q_values[:, i, 0, :].max(axis=-1).mean(axis=0)
                        for i in range(E)
                    }
                }
        assert target.shape == replay_chosen_q.shape

        if loss_type == 'huber':
            # Sum over ensemble, mean over batch
            # (B, E) or (B, E, E) to
            loss = losses.huber_loss(
                target, replay_chosen_q)
            info.update({})
        elif loss_type == 'mse':
            # (B, E) or (B, E, E) to
            loss = losses.mse_loss(
                target, replay_chosen_q)
        else:
            raise ValueError(f'Unknown loss type {loss_type}')

        # Bootstrapped masking
        if bootstrap:
            assert loss.shape == ens_mask.shape
            loss = loss * ens_mask
            # Equivalent to variable batch size for each ensemble member
            loss = loss.sum(axis=0) / ens_mask.sum(axis=0)
        else:
            # Mean over batch, (E) or (E, E)
            loss = loss.mean(axis=0)



        if aux_loss == 'none':
            # (E)
            info.update({
                f'loss_{i}': loss[i] for i in range(E)
            })
            # Sum over ensemble
            loss = loss.sum(axis=0)
        else:
            # (E, E)
            if not multi_gamma:
                info.update({
                    f'loss_{i}': loss[i, i] for i in range(E)
                })
            else:
                info.update({
                    f'loss_{i}': loss[i, 0] for i in range(E)
                })
            # (E, E) sum over ensemble and heads
            loss = loss.sum(axis=(0, 1))
        if mico:
            distance_fn = metric_utils.cosine_distance
            def q_target(state):
                return network_def.apply(target_params, state, share_batch=share_batch)
            target_r = jax.vmap(q_target, in_axes=(0))(states).representation
            if not network_def.share_encoder:
                assert target_r.shape[:2] == (B, E)
                assert target_r.ndim == 3
                online_dist = jax.vmap(metric_utils.representation_distances, in_axes=(1, 1, None), out_axes=-1)(
                    representations, target_r, distance_fn)
                target_dist = jax.vmap(metric_utils.target_distances, in_axes=(1, None, None, None), out_axes=-1)(
                    target_next_r, rewards, distance_fn, cumulative_gamma)
                metric_loss = losses.huber_loss(online_dist, target_dist)
                # (B, B, E)
                assert metric_loss.shape == (B*B, E), metric_loss.shape
                metric_loss = metric_loss.mean(axis=0).sum(axis=0)
            else:
                assert target_r.ndim == 2
                online_dist = metric_utils.representation_distances(
                    representations, target_r, distance_fn)
                target_dist = metric_utils.target_distances(
                    target_next_r, rewards, distance_fn, cumulative_gamma)
                metric_loss = losses.huber_loss(online_dist, target_dist)
                # (B, B, E)
                assert metric_loss.shape == (B*B,), metric_loss.shape
                metric_loss = metric_loss.mean(axis=0)

            loss = ((1. - mico_weight) * loss +
                    mico_weight * metric_loss)
            info.update({
                'mico_loss': metric_loss
            })
            # (E, E) sum over ensemble and heads
        return loss, info

    def q_online(state):
        return network_def.apply(online_params, state, share_batch=share_batch)

    def q_target(state):
        return network_def.apply(target_params, state, share_batch=share_batch)

    target, target_next_r = target_q(q_online, q_target, next_states, rewards, terminals,
                      cumulative_gamma, double_dqn, aux_loss, single_cerl,
                      multi_gamma, num_gammas, cerl_own_target)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, info), grad = grad_fn(online_params, target, target_next_r)

    # Scale gradient
    if grad_scale:
        grad = flax.core.unfreeze(grad)
        def scale_grad(key):
            grad['params'][key] = jax.tree_map(lambda x: x / E,
                                               grad['params'][key])
        if aux_loss == 'none':
            if network_def.share_encoder:
                scale_grad('Encoder')
            if network_def.share_penult:
                assert network_def.share_encoder
                scale_grad('Penult')

        elif aux_loss != 'none':
            assert aux_loss in ['final', 'both']
            if aux_loss == 'both':
                assert not network_def.share_encoder
                assert not network_def.share_penult
                scale_grad('Encoder')
            elif aux_loss == 'final':
                if not multi_gamma:
                    assert not network_def.share_penult
                    scale_grad('Encoder')
                    scale_grad('Penult')
                    if network_def.share_encoder:
                        # One more time 
                        scale_grad('Encoder')
                else:
                    # Hard coded
                    if E != 1:
                        assert not network_def.share_encoder
                    grad['params']['Encoder'] = jax.tree_map(lambda x: x / num_gammas,
                                                       grad['params']['Encoder'])
                    grad['params']['Penult'] = jax.tree_map(lambda x: x / num_gammas,
                                                       grad['params']['Penult'])
            else:
                raise ValueError(f'Invalid aux loss {aux_loss}')

        grad = flax.core.freeze(grad)

    updates, optimizer_state = optimizer.update(grad,
                                                optimizer_state,
                                                params=online_params)
    online_params = optax.apply_updates(online_params, updates)
    return optimizer_state, online_params, loss, info

def get_gammas(gamma_max: float, num_gammas: int, ensemble_size: int):
    assert num_gammas == 10, 'I hard-coded this just in case'
    h_max = 1.0 / (1 - gamma_max)
    # Large to small
    hs = jnp.array([(i + 1) * (h_max / num_gammas) for i in reversed(range(num_gammas))])
    # Large to small
    gammas = 1.0 - (1 / hs)
    # Stack (E, H)
    gammas = jnp.repeat(gammas[None, :], ensemble_size, axis=0)
    # Shift so the diganols have the largest gamma
    # rows, column_indices = jnp.mgrid[:num_gammas, :num_gammas]
    # column_indices -= jnp.arange(num_gammas)[:, None]
    # gammas = gammas[rows, column_indices]
    return gammas

def target_q(online_network, target_network, next_states, rewards, terminals,
             cumulative_gamma, double_dqn, aux_loss, single_cerl, multi_gamma,
             num_gammas, cerl_own_target):
    """Compute the target Q-value."""
    if double_dqn:
        # (B, E, A) or (B, E, E, A)
        next_state_q_vals_for_argmax = jax.vmap(
            online_network, in_axes=(0))(next_states).q_values
    else:
        # (B, E, A) or (B, E, E, A)
        next_state_q_vals_for_argmax = jax.vmap(
            target_network, in_axes=(0))(next_states).q_values
    (B, E) = next_state_q_vals_for_argmax.shape[:2]
    # next_state_q_vals_for_argmax = jnp.squeeze(next_state_q_vals_for_argmax)
    indices = jnp.arange(next_state_q_vals_for_argmax.shape[1])

    if aux_loss != 'none':
        if not multi_gamma:
            # (B, E, E, A) -> (B, E, A), diagonal elements
            # jnp.diagonal works with batch data
            next_state_q_vals_for_argmax = next_state_q_vals_for_argmax[:, indices,
                                                                            indices, :]
        else:
            # (B, E, H, A) -> (B, E, A)
            # First one
            next_state_q_vals_for_argmax = next_state_q_vals_for_argmax[:, :, 0, :]
    # (B, E, A) -> (B, E)
    next_argmax = jnp.argmax(next_state_q_vals_for_argmax, axis=-1)
    if single_cerl:
        # Only use the first one
        next_argmax = jnp.repeat(next_argmax[:, 0:1], repeats=next_argmax.shape[-1], axis=-1)
    # (B, E, A) or (B, E, E, A)
    output = jax.vmap(target_network, in_axes=(0))(next_states)
    q_values, target_next_r = output.q_values, output.representation
    if aux_loss == 'none':
        # (B, E, A), (B, E)
        replay_next_qt_max = jax.vmap(jax.vmap(lambda t, u: t[u]))(q_values,
                                                                   next_argmax)
    elif multi_gamma:
        # (B, E, H, A), (B, E) -> (B, E, H)
        replay_next_qt_max = jax.vmap(jax.vmap(lambda t, u: t[:, u]))(q_values,
                                                                   next_argmax)
    elif cerl_own_target:
        # Similar to CERL, except that the value comes from itself (but action
        # for argmax stills come from other agents)
        # Note the transpose is crucial
        # (B, E1, E2, A) (B, E2)
        # -> (B, E2, E1, A), (B, E2) -> (B, E2, E1)
        q_values = q_values.transpose(0, 2, 1, 3)
        replay_next_qt_max = jax.vmap(jax.vmap(lambda t, u: t[:, u]))(q_values,
                                                                   next_argmax)
        # Transpose  back
        # (B, E2, E1) -> (B, E1, E2)
        replay_next_qt_max = replay_next_qt_max.transpose(0, 2, 1)
    else:
        # (B, E, A)
        q_values = q_values[:, indices, indices, :]
        # (B, E)
        replay_next_qt_max = jax.vmap(jax.vmap(lambda t, u: t[u]))(q_values,
                                                                   next_argmax)
        # (B, 1, E): all ensemble uses the same set of target
        replay_next_qt_max = replay_next_qt_max[:, None, :].repeat(
            replay_next_qt_max.shape[-1], axis=1)

    # Batch last, for broadcasting with rewards and terminals
    # This is either (B, E) or (B, E, E), so it will become (E, B) or (E, E, B)
    replay_next_qt_max = jnp.moveaxis(replay_next_qt_max,
                                      source=0,
                                      destination=-1)
    # When not sharing batch, rewards is of shape (B, E). We change it to (E, B)
    # In this case replay_next_qt_max must be (B, E)
    # Otherwise this is non-op both rewards and terminals will be (B,)
    rewards = jnp.moveaxis(rewards, source=0, destination=-1)
    terminals = jnp.moveaxis(terminals, source=0, destination=-1)

    if multi_gamma:
        # (E, H)
        gammas = get_gammas(cumulative_gamma, num_gammas, replay_next_qt_max.shape[0])
        # (E, H, 1)
        gammas = gammas[:, :, None]
        assert gammas.shape == (E, num_gammas, 1)
        target = jax.lax.stop_gradient(rewards +
                                       gammas * replay_next_qt_max *
                                       (1. - terminals))
    else:
        target = jax.lax.stop_gradient(rewards +
                                       cumulative_gamma * replay_next_qt_max *
                                       (1. - terminals))
    # Move back
    target = jnp.moveaxis(target, source=-1, destination=0)
    return target, target_next_r


# @functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11))
@functools.partial(
    jax.jit,
    static_argnames=('network_def', 'num_actions', 'eval_mode', 'epsilon_eval',
                     'epsilon_train', 'epsilon_decay_period',
                     'min_replay_history', 'epsilon_fn', 'agent_id',
                     'aux_loss', 'network_def_single', 'multi_gamma'))
def select_action(network_def, params, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn, agent_id,
                  aux_loss, network_def_single, multi_gamma):
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
        q_values = network_def.apply(params, state, share_batch=True).q_values
        if aux_loss != 'none':
            indices = jnp.arange(q_values.shape[0])
            # (E, E, A) -> (E, A)
            if not multi_gamma:
                q_values = q_values[indices, indices, :]
            else:
                # (E, H, A) -> (E, A)
                q_values = q_values[:, 0, :]

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


        def extract(key, func):
            params['params'][key] = jax.tree_map(func, params['params'][key])

        if aux_loss == 'none':
            if not network_def.share_encoder:
                extract('Encoder', lambda x: x[agent_id])
            if not network_def.share_penult:
                extract('Penult', lambda x: x[agent_id])
            extract('Final', lambda x: x[agent_id][None])
        else:

            if aux_loss == 'final':
                assert not network_def.share_penult
                if not network_def.share_encoder:
                    extract('Encoder', lambda x: x[agent_id])
                extract('Penult', lambda x: x[agent_id])
            else:
                assert not network_def.share_encoder
                assert not network_def.share_penult
                assert aux_loss == 'both'
                extract('Encoder', lambda x: x[agent_id])
                extract('Penult', lambda x: x[agent_id, agent_id])
            if not multi_gamma:
                extract('Final', lambda x: x[agent_id, agent_id][None])
            else:
                # For some reason, the head params dim comes first
                extract('Final', lambda x: x[0, agent_id][None])

        params = flax.core.freeze(params)

        q_values = network_def_single.apply(params, state, share_batch=True).q_values
        # (1, A) -> (A)
        q_values = q_values.squeeze(axis=0)
        action = jnp.argmax(q_values, axis=0)
        entropy = None
        q_value = jnp.max(q_values, axis=0)  # for loggin

    info = {'q_value': q_value, 'entropy': entropy}

    return rng, jnp.where(p <= epsilon,
                          jax.random.randint(rng2, (), 0, num_actions),
                          action), info


@gin.configurable
class JaxBootDQNAgent(JaxDQNCustomAgent):
    def __init__(
        self,
        *args,
        ensemble_size: int,
        share_encoder: bool,
        share_penult: bool,
        aux_loss: str,
        tandem: bool,
        active_prob: Optional[tuple],
        grad_scale: bool,
        dqn_zoo_net: bool,
        sep_prob: Optional[float],  # If non, also use shared batches
        sep_mode: str,
        single_cerl: bool, # 
        multi_gamma: bool,
        num_gammas: Optional[int],
        cerl_own_target: bool,
        fnorm: bool,
        mico: bool,
        mico_weight: float,
        per_step: bool,
        boot_prob: Optional[float],
        **kwargs,
    ):
        """
        Arguments:
            share_encoder: share encoder across ensemble members
            share_penult: share penultimate layer across ensemble members
            aux_loss: in ['final', 'both', 'none']. 
                'final': use CERL loss. Duplicate last layer for CERL loss
                'both': use CERL loss. Duplicate the last twol ayers for CERL loss
                'none': not using cerl loss
            tandem: run tandem experiments. ensemble_size must be 2
            active_prob: the probability of using the active member for acting
            grad_scale: whether to scale gradients according to ensemble size
            dqn_zoo_net: use DQN Zoo network architecture and initialization
            seq_prob: if not None, use separate batches for different members with
                      this probability
            sep_mode: in ['own', 'disjoint', 'sym']:
                'own': different members only sample from the data they generated
                'disjoint': the data when entering the replay buffer are randomly 
                            assigned to one of the members. Each member only 
                            samples from the data they are assigned
                'sym': different members can sample from all the data
                         passive agent. This flag is not used
            single_cerl: CERL but only the first member is used for acting.
                         all other members learn the value of the first member
            multi_gamma: whether to use multi-horizon auxiliary loss
            num_gammas: as the name suggested
            cerl_own_target: similar to CERL, except that the value comes from 
                             itself (but actions  for argmax stills come from
                             other agents)
            fnorm: normalized the feature in the Q network. Not used
            mico: whether to use MICo
            mico_weight: as the name suggested
            per_step: at each step we randomly select one member for acting
            boot_prob: if not none, use bootstrapped sampling (it is called this
                       but we are not actually doing this.. See original Bootstrapped
                       DQN paper.)
        """
        assert sep_mode in ['own', 'disjoint', 'sym']
        assert aux_loss in ['none', 'final', 'both']
        if aux_loss != 'none':
            assert not share_penult
            if aux_loss == 'both':
                assert not share_encoder
        network_fn = functools.partial(NatureDQNNetworkEnsemble,
                                       ensemble_size=ensemble_size,
                                       share_encoder=share_encoder,
                                       share_penult=share_penult,
                                       aux_loss=aux_loss,
                                       dqn_zoo_net=dqn_zoo_net,
                                       multi_gamma=multi_gamma,
                                       num_gammas=num_gammas,
                                       fnorm=fnorm)
        # Needed for building replay buffer. Determine whether we log ensemble mask
        self.sep_prob = sep_prob
        self.sep_mode = sep_mode
        self.ensemble_size = ensemble_size
        self.boot_prob = boot_prob
        super().__init__(*args, network=network_fn, **kwargs)
        # Single network: for acting
        self.network_def_single = NatureDQNNetworkEnsemble(
            num_actions=self.num_actions,
            ensemble_size=1,
            share_encoder=True,
            share_penult=True,
            aux_loss='none',
            inputs_preprocessed=self.network_def.inputs_preprocessed,
            dqn_zoo_net=self.network_def.dqn_zoo_net,
            multi_gamma=False,  # this is deliberate
            num_gammas=None,
            fnorm=self.network_def.fnorm
        )
        self.share_encoder = share_encoder
        self.share_penult = share_penult
        self.aux_loss = aux_loss
        self.active_prob = active_prob
        self.tandem = tandem
        self.grad_scale = grad_scale
        self.dqn_zoo_net = dqn_zoo_net
        self.single_cerl = single_cerl
        self.multi_gamma = multi_gamma
        self.num_gammas = num_gammas
        if self.multi_gamma:
            assert self.num_gammas == 10
        self.cerl_own_target = cerl_own_target
        self.mico = mico
        self.mico_weight = mico_weight
        self.per_step = per_step
        if active_prob is not None:
            assert self.tandem
        logging.info('\t ensemble_size: %s', self.ensemble_size)
        logging.info('\t share_encoder: %s', self.share_encoder)
        logging.info('\t share_penult: %s', self.share_penult)
        logging.info('\t aux_loss: %s', self.aux_loss)
        logging.info('\t active_prob: %s', self.active_prob)
        logging.info('\t tandem: %s', self.tandem)
        logging.info('\t grad_scale: %s', self.grad_scale)
        logging.info('\t dqn_zoo_net: %s', self.dqn_zoo_net)
        logging.info('\t sep_prob: %s', self.sep_prob)
        logging.info('\t sep_mode: %s', self.sep_mode)
        logging.info('\t single_cerl: %s', self.single_cerl)
        logging.info('\t multi_gamma: %s', self.multi_gamma)
        logging.info('\t cerl_own_target: %s', self.cerl_own_target)
        logging.info('\t mico: %s', self.mico)
        logging.info('\t mico_weight: %s', self.mico_weight)
        logging.info('\t per_step: %s', self.per_step)
        logging.info('\t boot_prob: %s', self.boot_prob)
        self.current_agent_id = 0
        # one of ['vote', 'random', 'single']
        # random: for each episode, randomly select one. proxy for average performance
        # single specific agent
        self.eval_policy = 'vote'
        self.eval_agent_id = None

        # Log entropy
        self.entropy_list = []

    def _store_transition(self,
                          last_observation,
                          action,
                          reward,
                          is_terminal,
                          *args,
                          priority=None,
                          episode_end=False):
        assert not self.eval_mode
        assert len(args) == 0
        if self.sep_prob is not None and self.sep_mode != 'sym':
            if self.sep_mode == 'own':
                ens_mask = (onp.arange(self.ensemble_size) == self.current_agent_id)
            elif self.sep_mode == 'disjoint':
                ens_mask = (onp.arange(self.ensemble_size) == onp.random.choice(self.ensemble_size))
            else:
                raise ValueError(f'Invalid sample mode {self.sep_mode}')
            args = (ens_mask,)
        elif self.boot_prob is not None:
            ens_mask = onp.random.rand(self.ensemble_size) < self.boot_prob
            args = (ens_mask,)
        else:
            args = ()
        super()._store_transition(last_observation,
                                  action,
                                  reward,
                                  is_terminal,
                                  *args,
                                  priority=priority,
                                  episode_end=episode_end)
    
    def _build_networks_and_optimizer(self):
        self._rng, rng = jax.random.split(self._rng)
        # The purpose of this is just to make share batch explicit
        self.online_params = self.network_def.init(rng, x=self.state, share_batch=True)
        logging.info('Number of parameters: {}'.format(sum(jnp.prod(jnp.array(x.shape)) for x in jax.tree_util.tree_leaves(self.online_params))))

        print(jax.tree_map(lambda x: x.shape, self.online_params))
        self.optimizer = create_optimizer(self._optimizer_name)
        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = self.online_params

    def _build_replay_buffer(self):
        """Creates the replay buffer used by the agent."""
        if self.sep_prob is not None and self.sep_mode != 'sym':
            extra_storage_types = [ReplayElement('ens_mask', (self.ensemble_size,), onp.bool_)]
        elif self.boot_prob is not None:
            extra_storage_types = [ReplayElement('ens_mask', (self.ensemble_size,), onp.bool_)]
        else:
            extra_storage_types = None
        return VectorizedOutOfGraphReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype,
            extra_storage_types=extra_storage_types
        )

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

    # def true_begin_episode(self):
        # begin_episode will also be called when life is lost. 
        # self.set_acting_agent_id()

    def set_acting_agent_id(self):
        if self.eval_mode:
            if self.eval_policy == 'random':
                self.eval_agent_id = onp.random.choice(self.ensemble_size)
        else:
            if self.tandem:
                assert self.active_prob is not None
                assert self.ensemble_size == 2
                self.current_agent_id = onp.random.choice(
                    self.ensemble_size,
                    p=[self.active_prob, 1.0 - self.active_prob])
            else:
                self.current_agent_id = onp.random.choice(
                    self.ensemble_size)

    def get_acting_agent_id(self):
        if self.single_cerl:
            return 0
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

    def _sample_from_replay_buffer(self):
        if self.sep_prob is None or onp.random.random() >= self.sep_prob:
            samples = self._replay.sample_transition_batch()
        else:
            samples = self._replay.sample_transition_multi_batch(num_batches=self.ensemble_size, asym=(self.sep_mode != 'sym'))
        types = self._replay.get_transition_elements()
        self.replay_elements = collections.OrderedDict()
        for element, element_type in zip(samples, types):
            self.replay_elements[element_type.name] = element

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

                if self.boot_prob is None:
                    # Actually this is not even used
                    ens_mask = None
                else:
                    ens_mask = self.replay_elements['ens_mask']

                self.optimizer_state, self.online_params, loss, info = train(
                    self.network_def,
                    self.online_params,
                    self.target_network_params,
                    self.optimizer,
                    self.optimizer_state,
                    states,
                    self.replay_elements['action'],
                    next_states,
                    self.replay_elements['reward'],
                    self.replay_elements['terminal'],
                    self.cumulative_gamma,
                    loss_type=self._loss_type,
                    double_dqn=self.double_dqn,
                    aux_loss=self.aux_loss,
                    grad_scale=self.grad_scale,
                    single_cerl=self.single_cerl,
                    multi_gamma=self.multi_gamma,
                    num_gammas=self.num_gammas,
                    cerl_own_target=self.cerl_own_target,
                    mico=self.mico,
                    mico_weight=self.mico_weight,
                    ens_mask=ens_mask,
                    bootstrap=(self.boot_prob is not None)
                )


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
            aux_loss=self.aux_loss,
            network_def_single=self.network_def_single,
            multi_gamma=self.multi_gamma
        )
        if self.eval_mode:
            entropy = info.pop('entropy')
            assert entropy is not None
            entropy = entropy.item()
            self.entropy_list.append(entropy)
        self.action = onp.asarray(self.action)
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

        if self.per_step and not self.eval_mode:
            # Switch ensemble on a per-step basis during training
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
            aux_loss=self.aux_loss,
            network_def_single=self.network_def_single,
            multi_gamma=self.multi_gamma
        )

        if self.eval_mode:
            entropy = info.pop('entropy')
            assert entropy is not None
            entropy = entropy.item()
            self.entropy_list.append(entropy)
        self.action = onp.asarray(self.action)
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

