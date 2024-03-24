from dopamine.jax.networks import preprocess_atari_inputs, QuantileNetwork
import flax.linen as nn
import gin
from dopamine.discrete_domains import atari_lib
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import functools
import collections


# From dqn_zoo
def _dqn_default_initializer(num_input_units: int,
                             dtype=jnp.float32) -> nn.initializers.Initializer:
    """Default initialization scheme inherited from past implementations of DQN.
      This scheme was historically used to initialize all weights and biases
      in convolutional and linear layers of DQN-type agents' networks.
      It initializes each weight as an independent uniform sample from [`-c`, `c`],
      where `c = 1 / np.sqrt(num_input_units)`, and `num_input_units` is the number
      of input units affecting a single output unit in the given layer, i.e. the
      total number of inputs in the case of linear (dense) layers, and
      `num_input_channels * kernel_width * kernel_height` in the case of
      convolutional layers.
      Args:
        num_input_units: number of input units to a single output unit of the layer.
      Returns:
        Haiku weight initializer.
      """
    max_val = jnp.sqrt(1 / num_input_units)

    def init(key, shape, dtype=dtype):
        return jax.random.uniform(key=key,
                                  shape=shape,
                                  dtype=dtype,
                                  minval=-max_val,
                                  maxval=max_val)

    return init


class DQNZooConv(nn.Module):
    features: int
    kernel_size: Tuple[int]
    strides: Tuple[int]

    # Conv with dqn_zoo init
    @nn.compact
    def __call__(self, x):
        num_input_units = x.shape[-1] * self.kernel_size[0] * self.kernel_size[
            1]
        initializer = _dqn_default_initializer(num_input_units)
        x = nn.Conv(features=self.features,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    kernel_init=initializer,
                    bias_init=initializer,
                    padding='VALID')(x)
        return x


class DQNZooDense(nn.Module):
    features: int

    # Conv with dqn_zoo init
    @nn.compact
    def __call__(self, x):
        num_input_units = x.shape[-1]
        initializer = _dqn_default_initializer(num_input_units)
        x = nn.Dense(features=self.features,
                     kernel_init=initializer,
                     bias_init=initializer)(x)
        return x


class DQNZooDenseWithSharedBias(nn.Module):
    features: int

    # Conv with dqn_zoo init
    @nn.compact
    def __call__(self, x):
        num_input_units = x.shape[-1]
        initializer = _dqn_default_initializer(num_input_units)
        x = nn.Dense(features=self.features,
                     use_bias=False,
                     kernel_init=initializer)(x)
        bias = self.param('bias', initializer, (1, ))
        bias = jnp.broadcast_to(bias, x.shape)
        return x + bias


class NatureDQNEncoder(nn.Module):
    """The convolutional network used to compute the agent's Q-values."""
    inputs_preprocessed: bool = False

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        if not self.inputs_preprocessed:
            x = preprocess_atari_inputs(x)
        x = nn.Conv(features=32,
                    kernel_size=(8, 8),
                    strides=(4, 4),
                    kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    kernel_init=initializer)(x)
        x = nn.relu(x)
        x = x.reshape((-1))  # flatten
        return x


class DQNZooNatureDQNEncoder(nn.Module):
    """The convolutional network used to compute the agent's Q-values."""
    inputs_preprocessed: bool = False

    @nn.compact
    def __call__(self, x):
        if not self.inputs_preprocessed:
            x = preprocess_atari_inputs(x)
        x = DQNZooConv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = DQNZooConv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = DQNZooConv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((-1))  # flatten
        return x


class NatureDQNHead(nn.Module):
    """The convolutional network used to compute the agent's Q-values."""
    num_actions: int

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        x = nn.Dense(features=512, kernel_init=initializer)(x)
        x = nn.relu(x)
        q_values = nn.Dense(features=self.num_actions,
                            kernel_init=initializer)(x)
        return q_values


class DQNZooNatureDQNHead(nn.Module):
    """The convolutional network used to compute the agent's Q-values."""
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = DQNZooDense(features=512)(x)
        x = nn.relu(x)
        q_values = DQNZooDenseWithSharedBias(features=self.num_actions)(x)
        return q_values

DQNNetworkTypeWithRepresentation = collections.namedtuple('dqn_network', ['q_values', 'representation'])

@gin.configurable
class NatureDQNNetworkEnsemble(nn.Module):
    """The convolutional network used to compute the agent's Q-values.

    Arguments:
        share_encoder: whether to share encoder across ensemble members
        share_penult: whether to share the penultimate layer
        aux_loss: whether to "branch" the network for auxiliary losses. `final`
            means only the final layer is duplicated. `both` means the top
            two layers will be duplicated
        dqn_zoo_net: whether to use the architecture in DQN zoo. This is faster
            because the hidden size is smaller
        multi_gamma: use multi-horizon auxiliary loss
        num_gamma: number of gammas, if multi_gamma=True
            
    """
    num_actions: int
    share_encoder: bool
    share_penult: bool
    ensemble_size: int
    aux_loss: str  # in ['final', 'both', 'none']
    dqn_zoo_net: bool
    multi_gamma: bool
    num_gammas: Optional[int]
    inputs_preprocessed: bool = False
    penult_size: int = 512
    fnorm: bool = False
    fn_eps: float = 1e-6

    @nn.compact
    def __call__(self, x, share_batch):
        if not share_batch:
            assert self.aux_loss == 'none'
        if self.share_penult:
            assert self.share_encoder
        if self.dqn_zoo_net:
            EncoderClass = DQNZooNatureDQNEncoder
            PenultClass = DQNZooDense
            FinalClass = DQNZooDenseWithSharedBias
        else:
            EncoderClass = NatureDQNEncoder
            PenultClass = nn.Dense
            FinalClass = nn.Dense

        if not self.share_encoder or not share_batch:
            EncoderClass = nn.vmap(
                EncoderClass,
                in_axes=None if share_batch else 0,  # shared batch
                out_axes=0,  # shared batch
                variable_axes={'params': 0} if not self.share_encoder else {'params': None},
                split_rngs={'params': True} if not self.share_encoder else {'params': False},
                axis_size=self.ensemble_size)

        if not self.share_penult or not share_batch:
            PenultClass = nn.vmap(
                PenultClass,
                in_axes=None if self.share_encoder and share_batch else 0,  # shared batch
                out_axes=0,  # shared batch
                variable_axes={'params': 0} if not self.share_penult else {'params': None},
                split_rngs={'params': True} if not self.share_penult else {'params': False},
                axis_size=self.ensemble_size)
        FinalClass = nn.vmap(
            FinalClass,
            in_axes=None if self.share_penult and share_batch else 0,  # shared batch
            out_axes=0,  # shared batch
            variable_axes={'params': 0},
            split_rngs={'params': True},
            axis_size=self.ensemble_size)


        if self.multi_gamma:
            assert self.aux_loss == 'final' and self.num_gammas == 10
            if self.ensemble_size != 1:
                assert not self.share_encoder and not self.share_penult
        else:
            assert self.num_gammas is None

        if self.aux_loss != 'none':
            # assert not self.share_encoder
            if self.aux_loss == 'final':
                assert not self.share_penult
                # 10 ensemble members, each having 10 heads
                num_heads = self.ensemble_size if not self.multi_gamma else self.num_gammas
                FinalClass = nn.vmap(
                    FinalClass,
                    in_axes=None,  # shared batch
                    out_axes=1,  # shared batch
                    variable_axes={'params': 0},
                    split_rngs={'params': True},
                    axis_size=num_heads)
            elif self.aux_loss == 'both':
                assert not self.share_encoder
                assert not self.share_penult
                PenultClass = nn.vmap(
                    PenultClass,
                    in_axes=None,  # shared batch
                    out_axes=1,  # shared batch
                    variable_axes={'params': 0},
                    split_rngs={'params': True},
                    axis_size=self.ensemble_size)
                FinalClass = nn.vmap(
                    FinalClass,
                    in_axes=1,  # shared batch
                    out_axes=1,  # shared batch
                    variable_axes={'params': 0},
                    split_rngs={'params': True},
                    axis_size=self.ensemble_size)
            else:
                raise ValueError()
        encoder = EncoderClass(inputs_preprocessed=self.inputs_preprocessed, name='Encoder')
        if not self.dqn_zoo_net:
            initializer = nn.initializers.xavier_uniform()
            penult = PenultClass(features=self.penult_size, kernel_init=initializer, name='Penult')
            final = FinalClass(features=self.num_actions,
                               kernel_init=initializer, name='Final')
        else:
            # DQN zoo has init built-in
            penult = PenultClass(features=self.penult_size, name='Penult')
            final = FinalClass(features=self.num_actions, name='Final')

        x = encoder(x)
        representation = x
        x = penult(x)
        x = nn.relu(x)
        if self.fnorm:
            l2_norm = jnp.linalg.norm(x, axis=-1, ord=2, keepdims=True)
            # Feature normalization
            x = x / (l2_norm + self.fn_eps)
        q_values = final(x)
        # This could be (N, *, A) or (N, N, *, A)
        # q_values = head(x)
        return DQNNetworkTypeWithRepresentation(q_values, representation)

@gin.configurable
class QuantileNetworkEnsemble(nn.Module):
  """Convolutional network used to compute the agent's return quantiles."""
  num_actions: int
  num_atoms: int
  inputs_preprocessed: bool = False
  ensemble_size: int = 1

  @nn.compact
  def __call__(self, x):

    EnsembleClass = nn.vmap(
        QuantileNetwork,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=0,
        axis_size=self.ensemble_size
    )
    ensemble = EnsembleClass(
        num_actions=self.num_actions,
        num_atoms=self.num_atoms,
        inputs_preprocessed=self.inputs_preprocessed
    )
    return ensemble(x)
