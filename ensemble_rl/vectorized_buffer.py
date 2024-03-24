from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer, STORE_FILENAME_PREFIX
import numpy as np
import tensorflow as tf
import gzip
import pickle
import gin


@gin.configurable
class VectorizedOutOfGraphReplayBuffer(OutOfGraphReplayBuffer):
    def __init__(self, *args, checkpoint_frequency, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_frequency = checkpoint_frequency

    """Vectorized version of the original slow-as-hell buffer"""
    def get_invalid_indices(self, indices):
        """Checks if the indices contains a valid transition.

        Checks for collisions with the end of episodes and the current position
        of the cursor.

        Args:
          indices: int, the indices to the state in the transition.

        Returns:
            set of indices with invalid transitions

        """
        invalid_mask = np.zeros_like(indices, dtype=bool)

        # Check the indices is in the valid range
        invalid_mask |= (indices < 0) & (indices >= self._replay_capacity)
        if not self.is_full():
            invalid_mask |= (indices >= self.cursor() - self._update_horizon)
            invalid_mask |= (indices < self._stack_size - 1)
            #     return False

        # Skip transitions that straddle the cursor.
        invalid_mask |= np.isin(indices, self.invalid_range)

        # If there are terminal flags in any other frame other than the last one
        # the stack is not valid, so don't sample it.
        stack_offset = np.arange(self._stack_size)[::-1][None, :]
        stack_indices = (indices[:, None] -
                         stack_offset) % self._replay_capacity
        invalid_mask |= self._store['terminal'][stack_indices][:, :-1].any(
            axis=1)

        # If the episode ends before the update horizon, without a terminal signal,
        # it is invalid.
        trajectory_indices = (indices[:, None] + np.arange(
            self._update_horizon)[None, :]) % self._replay_capacity
        end_indices = np.array(list(self.episode_end_indices))
        # Truncated: episode end but not terminal

        # Note that terminal is uint8, and & and ~ and actually bitwise operators
        # which means ~(np.array(1, dtype=np.uint8)) is actually 254.
        # However, it so happens that the results are correct. You can verify
        # the truth table.
        # I should have used np.logical_not and np.logical_and instead though.
        truncated = np.isin(trajectory_indices, end_indices) & (
            ~self._store['terminal'][trajectory_indices])
        invalid_mask |= truncated.any(axis=1)

        return np.flatnonzero(invalid_mask)

    def sample_index_batch_vectorized(self, batch_size, agent_id=None):
        """Returns a batch of valid indices sampled uniformly.

        Args:
          batch_size: int, number of indices returned.

        Returns:
          list of ints, a batch of valid indices sampled uniformly.

        Raises:
          RuntimeError: If the batch was not constructed after maximum number of
            tries.
        """
        if self.is_full():
            # add_count >= self._replay_capacity > self._stack_size
            min_id = self.cursor(
            ) - self._replay_capacity + self._stack_size - 1
            max_id = self.cursor() - self._update_horizon
        else:
            # add_count < self._replay_capacity
            min_id = self._stack_size - 1
            max_id = self.cursor() - self._update_horizon
            if max_id <= min_id:
                raise RuntimeError(
                    'Cannot sample a batch with fewer than stack size '
                    '({}) + update_horizon ({}) transitions.'.format(
                        self._stack_size, self._update_horizon))

        attempt_count = 0

        def get_index_sym(size):
            # Symmetric: same stuff regardless of agent id
            indices = np.random.randint(min_id, max_id,
                                        size=size) % self._replay_capacity
            return indices

        def get_index_asym(size):
            # Asymmetric: result is agent id dependent
            ens_mask = self._store['ens_mask'][min_id:max_id, agent_id]
            assert ens_mask.ndim == 1
            non_zero_indices = np.flatnonzero(ens_mask)
            count = non_zero_indices.shape[0]
            if count > 0:
                indices_within_agent = np.random.randint(count, size=batch_size)
                indices = non_zero_indices[indices_within_agent]
                # Offsete
                indices += min_id
                indices = indices % self._replay_capacity
                return indices
            else:
                # The unlikely case of no enough transitions for a particular member
                # we fall back to symmetric sampling
                return get_index_sym(size)

        if agent_id is None:
            sample_func = get_index_sym
        else:
            sample_func = get_index_asym

        indices = sample_func(batch_size)
        while attempt_count < self._max_sample_attempts:
            invalid_indices = self.get_invalid_indices(indices)

            # Sanity check
            # invalid_original = []
            # for pos, index in enumerate(indices):
            #     if not self.is_valid_transition(index):
            #         invalid_original.append(pos)
            # invalid_original = np.array(invalid_original)
            # assert invalid_indices.shape == invalid_original.shape
            # assert np.all(invalid_indices == invalid_original)


            if len(invalid_indices) == 0:
                break
            else:
                # indices[invalid_indices] = np.random.randint(
                    # min_id, max_id,
                    # size=len(invalid_indices)) % self._replay_capacity
                indices[invalid_indices] = sample_func(len(invalid_indices))
                attempt_count += 1
        else:
            raise RuntimeError(
                'Max sample attempts: Tried {} times. Batch size is {}'.format(
                    self._max_sample_attempts, batch_size))

        return indices

    def sample_transition_multi_batch(self, num_batches: int, asym: bool):
        B = self._batch_size
        E = num_batches
        if asym:
            indices = [self.sample_index_batch_vectorized(self._batch_size, agent_id=agent_id) for agent_id in range(num_batches)]
            # (B, E) -> (B*E,)
            indices = np.stack(indices, axis=1).ravel()
        else:
            indices = self.sample_index_batch_vectorized(self._batch_size * num_batches)
        batch_size = self._batch_size * num_batches
        batch_arrays = self.sample_transition_batch(batch_size=batch_size, indices=indices)

        # Unravel, (B*E, *) -> (B, E, *)
        assert all((array.shape[0] == B * E) for array in batch_arrays)
        batch_arrays = [array.reshape(B, E, *array.shape[1:]) for array in batch_arrays]
        return batch_arrays

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions (including any extra contents).

        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        OutOfGraphPrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.

        When the transition is terminal next_state_batch has undefined contents.

        NOTE: This transition contains the indices of the sampled elements. These
        are only valid during the call to sample_transition_batch, i.e. they may
        be used by subclasses of this replay buffer but may point to different data
        as soon as sampling is done.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().

        Raises:
          ValueError: If an element to be sampled is missing from the replay buffer.
        """
        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            # (B,)
            indices = self.sample_index_batch_vectorized(batch_size)
        assert len(indices) == batch_size

        # Get next state indices because it could be terminal
        indices = np.array(indices, dtype=np.int32)
        # (B, T) = (B, 1) + (1, T)
        trajectory_indices = (indices[:, None] + np.arange(
            self._update_horizon)[None, :]) % self._replay_capacity
        # (B, T)
        trajectory_terminals = self._store['terminal'][trajectory_indices, ...]
        # (B)
        is_terminal_transition = trajectory_terminals.any(axis=1)
        # (B)
        trajectory_length = np.where(
            is_terminal_transition,
            np.argmax(trajectory_terminals.astype(bool), axis=1) + 1,
            self._update_horizon)
        # (B)
        next_indices = (indices + trajectory_length) % self._replay_capacity

        # Get stack indices
        # (B, 1) - (1, S)
        # offset [3, 2, 1, 0]
        stack_offset = np.arange(self._stack_size)[::-1][None, :]
        stack_indices = (indices[:, None] -
                         stack_offset) % self._replay_capacity
        next_stack_indices = (next_indices[:, None] -
                              stack_offset) % self._replay_capacity

        # Fill the contents of each array in the sampled batch.
        transition_elements = self.get_transition_elements(batch_size)
        # batch_arrays = self._create_batch_arrays(batch_size)
        batch_arrays = []
        # (B, T) = (B, 1) + (1, T)
        # assert len(transition_elements) == len(batch_arrays)
        for element in transition_elements:
            if element.name == 'state':
                # (B, S, H, W)
                output = self._store['observation'][stack_indices]
                # (B, H, W, S)
                output = np.moveaxis(output, source=1, destination=-1)
            elif element.name == 'reward':
                # compute the discounted sum of rewards in the trajectory.
                # (B, T) = (1, T) < (B, 1)
                reward_mask = np.arange(
                    self._update_horizon)[None, :] < trajectory_length[:, None]
                # (B, T)
                trajectory_rewards = self._store['reward'][trajectory_indices]
                # (B, T) * (T) * (B, T)
                # (B, T) -> (B)
                output = np.sum(trajectory_rewards *
                                self._cumulative_discount_vector * reward_mask,
                                axis=1)
            elif element.name == 'next_state':
                # (B, S, H, W)
                output = self._store['observation'][next_stack_indices]
                # (B, H, W, S)
                output = np.moveaxis(output, source=1, destination=-1)
            elif element.name in ('next_action', 'next_reward'):
                output = (
                    self._store[element.name.lstrip('next_')][next_indices])
            elif element.name == 'terminal':
                output = is_terminal_transition
            elif element.name == 'indices':
                output = indices
            elif element.name in self._store.keys():
                output = self._store[element.name][indices]
            else:
                raise ValueError(f'Invalid element {element.name}')
            # We assume the other elements are filled in by the subclass.
            assert output.shape == element.shape
            if output.dtype != element.type:
                # astype copies by default
                output = output.astype(element.type)
            batch_arrays.append(output)

        # Sanity check
        # batch_original = super().sample_transition_batch(indices=indices)
        # assert len(batch_arrays) == len(batch_original)
        # for vec, original in zip(batch_arrays, batch_original):
        #     assert vec.dtype == original.dtype
        #     assert vec.shape == original.shape
        #     assert np.all(vec == original)

        return tuple(batch_arrays)

    def save(self, checkpoint_dir, iteration_number):
        """Save the OutOfGraphReplayBuffer attributes into a file.

        Handle checkpoint frequency
      This method will save all the replay buffer's state in a single file.

      Args:
        checkpoint_dir: str, the directory where numpy checkpoint files should be
          saved.
        iteration_number: int, iteration_number to use as a suffix in naming
          numpy checkpoint files.
      """
        if iteration_number % self.checkpoint_frequency != 0:
            return
        if not tf.io.gfile.exists(checkpoint_dir):
            return

        checkpointable_elements = self._return_checkpointable_elements()

        for attr in checkpointable_elements:
            filename = self._generate_filename(checkpoint_dir, attr,
                                               iteration_number)
            with tf.io.gfile.GFile(filename, 'wb') as f:
                with gzip.GzipFile(fileobj=f, mode='wb') as outfile:
                    # Checkpoint the np arrays in self._store with np.save instead of
                    # pickling the dictionary is critical for file size and performance.
                    # STORE_FILENAME_PREFIX indicates that the variable is contained in
                    # self._store.
                    if attr.startswith(STORE_FILENAME_PREFIX):
                        array_name = attr[len(STORE_FILENAME_PREFIX):]
                        np.save(outfile,
                                self._store[array_name],
                                allow_pickle=False)
                    # Some numpy arrays might not be part of storage
                    elif isinstance(self.__dict__[attr], np.ndarray):
                        np.save(outfile,
                                self.__dict__[attr],
                                allow_pickle=False)
                    else:
                        pickle.dump(self.__dict__[attr], outfile)

            # After writing a checkpoint file, we garbage collect the checkpoint file
            # that is four versions old.
            stale_iteration_number = iteration_number - self._checkpoint_duration * self.checkpoint_frequency

            # If keep_every has been set, we spare every keep_every'th checkpoint.
            if (self._keep_every is not None
                    and (stale_iteration_number % self._keep_every == 0)):
                return

            if stale_iteration_number >= 0:
                stale_filename = self._generate_filename(
                    checkpoint_dir, attr, stale_iteration_number)
                try:
                    tf.io.gfile.remove(stale_filename)
                except tf.errors.NotFoundError:
                    pass
