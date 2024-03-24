from dopamine.discrete_domains import run_experiment
import gin

# from dopamine.jax.agents.dqn import dqn_agent as jax_dqn_agent
from ensemble_rl.agents.dqn_agent import JaxDQNCustomAgent
from ensemble_rl.agents.boot_dqn_agent import JaxBootDQNAgent
from ensemble_rl.agents.boot_quantile_agent import JaxBootQuantileAgent
from dopamine.metrics import statistics_instance
from dopamine.discrete_domains import iteration_statistics, checkpointer
import logging
import wandb
import numpy as np
from dopamine.metrics import collector_dispatcher
from ensemble_rl.json_collector import JSONCollector
import shutil
from pathlib import Path
import sys
from collections import defaultdict
from datetime import datetime
import uuid


@gin.configurable
def create_agent_custom(sess,
                        environment,
                        agent_name=None,
                        summary_writer=None,
                        debug_mode=False):
    """Creates an agent.

  Args:
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
    assert agent_name is not None
    if not debug_mode:
        summary_writer = None
    if agent_name == 'dqn':
        return JaxDQNCustomAgent(num_actions=environment.action_space.n,
                                 summary_writer=summary_writer)
    elif agent_name == 'boot_dqn':
        return JaxBootDQNAgent(num_actions=environment.action_space.n,
                               summary_writer=summary_writer)
    elif agent_name == 'boot_quantile':
        return JaxBootQuantileAgent(num_actions=environment.action_space.n,
                               summary_writer=summary_writer)
    else:
        raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
class EnsembleRunner(run_experiment.Runner):
    def __init__(self, *args, use_legacy_logger=False, save_checkpoint=True, clean_final_checkpoint=False, **kwargs):
        # Register json logging
        assert not use_legacy_logger
        collector_dispatcher.add_collector('json', JSONCollector)
        self.collector_data = None
        super().__init__(*args, use_legacy_logger=use_legacy_logger, **kwargs)
        # resume is called before creating collector, so we do it here
        if self.collector_data is not None:
            for collector in self._collector_dispatcher._collectors:
                if collector.get_name() in self.collector_data:
                    assert hasattr(collector, 'load')
                    collector.load(self.collector_data[collector.get_name()])

        self.save_checkpoint = save_checkpoint
        self.clean_final_checkpoint = clean_final_checkpoint
        # self.stats_by_agent = defaultdict(lambda: defaultdict(list))
        self.stats_by_agent = None

    def _initialize_checkpointer_and_maybe_resume(self,
                                                  checkpoint_file_prefix):
        """Reloads the latest checkpoint if it exists.

      This method will first create a `Checkpointer` object and then call
      `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
      checkpoint in self._checkpoint_dir, and what the largest file number is.
      If a valid checkpoint file is found, it will load the bundled data from this
      file and will pass it to the agent for it to reload its data.
      If the agent is able to successfully unbundle, this method will verify that
      the unbundled data contains the keys,'logs' and 'current_iteration'. It will
      then load the `Logger`'s data from the bundle, and will return the iteration
      number keyed by 'current_iteration' as one of the return values (along with
      the `Checkpointer` object).

      Args:
        checkpoint_file_prefix: str, the checkpoint file prefix.

      Returns:
        start_iteration: int, the iteration number to start the experiment from.
        experiment_checkpointer: `Checkpointer` object for the experiment.
      """
        # complete_path = Path(self._base_dir) / 'complete.txt'
        # if complete_path.is_file():
        #     logging.info(
        #         f'WARNING: {complete_path} exists. run seems to be completed. Exiting...'
        #     )
        #     sys.exit()

        self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                       checkpoint_file_prefix)
        self._start_iteration = 1
        # Check if checkpoint exists. Note that the existence of checkpoint 0 means
        # that we have finished iteration 1 (so we will start from iteration 2).
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
            self._checkpoint_dir)
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(
                latest_checkpoint_version)
            if self._agent.unbundle(self._checkpoint_dir,
                                    latest_checkpoint_version,
                                    experiment_data):
                if experiment_data is not None:
                    # assert 'logs' in experiment_data
                    assert 'current_iteration' in experiment_data
                    assert 'collector_data' in experiment_data
                    # if self._use_legacy_logger:
                    # self._logger.data = experiment_data['logs']
                    self._start_iteration = experiment_data[
                        'current_iteration'] + 1
                    self.collector_data = experiment_data['collector_data']
                logging.info(
                    'Reloaded checkpoint and will start from iteration %d',
                    self._start_iteration)
                resume_info_path = Path(self._base_dir) / 'resume_info.txt'
                if resume_info_path.is_file():
                    # Save old resume info if it exists
                    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    archive_resume_info_path = Path(self._base_dir) / f'resume_info_{date_str}_{unique_id}.txt'
                    resume_info_path.rename(archive_resume_info_path)
                with resume_info_path.open('w') as f:
                    f.write(str(self._start_iteration))
            else:
                raise Exception('Loading checkpoint failed.')


    def _checkpoint_experiment(self, iteration):
        """Checkpoint experiment data.

        Args:
          iteration: int, iteration number for checkpointing.
        """
        if iteration % self._checkpointer._checkpoint_frequency != 0:
            # Stupid stuff because agent will save buffer
            return
        experiment_data = self._agent.bundle_and_checkpoint(
            self._checkpoint_dir, iteration)
        if experiment_data:
            experiment_data['current_iteration'] = iteration
            # if self._use_legacy_logger:
            # experiment_data['logs'] = self._logger.data
            collector_data = {}
            for collector in self._collector_dispatcher._collectors:
                if hasattr(collector, 'dump'):
                    collector_data[collector.get_name()] = collector.dump()
            experiment_data['collector_data'] = collector_data
            self._checkpointer.save_checkpoint(iteration, experiment_data)

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction.

        An iteration involves running several episodes until a certain number of
        steps are obtained. The interleaving of train/eval phases implemented here
        are to match the implementation of (Mnih et al., 2015).

        Args:
          iteration: int, current iteration number, used as a global_step for saving
            Tensorboard summaries.

        Returns:
        A dict containing summary statistics for this iteration.
        """
        statistics = iteration_statistics.IterationStatistics()
        logging.info('Starting iteration %d', iteration)
        self.reset_stats_by_agent()
        num_episodes_train, average_reward_train, average_steps_per_second = (
            self._run_train_phase(statistics))
        train_stats_by_agent = self.get_stats_by_agent()

        if self._agent.tandem:
            # Two evaluations, one passive one active
            assert self._agent.ensemble_size == 2

            self.reset_stats_by_agent()
            self._agent.reset_entropy_list()
            self._agent.set_eval_policy(eval_policy='single', eval_agent_id=0)
            num_episodes_active, average_reward_active = self._run_eval_phase(
                statistics)
            entropy_active = np.mean(self._agent.entropy_list)
            eval_stats_by_agent_active = self.get_stats_by_agent()

            self.reset_stats_by_agent()
            self._agent.reset_entropy_list()
            self._agent.set_eval_policy(eval_policy='single', eval_agent_id=1)
            num_episodes_passive, average_reward_passive = self._run_eval_phase(
                statistics)
            entropy_passive = np.mean(self._agent.entropy_list)
            eval_stats_by_agent_passive = self.get_stats_by_agent()

            # They won't overlap anyways
            eval_stats_by_agent = {**eval_stats_by_agent_active, **eval_stats_by_agent_passive}

            # I know this is not exactly very well organized...
            stats = {
                'iteration': iteration,
                'Train/NumEpisodes': num_episodes_train,
                'Train/AverageReturns': average_reward_train,
                'Train/AverageStepsPerSecond': average_steps_per_second,
                **{
                    f'Train/ByAgent/AverageReturns_{i}': train_stats_by_agent[i]['average_returns']
                    for i in train_stats_by_agent
                },
                **{
                    f'Train/ByAgent/NumEpisodes_{i}': train_stats_by_agent[i]['num_episodes']
                    for i in train_stats_by_agent
                },
                **{
                    f'Train/ByAgent/AverageQValues_{i}': train_stats_by_agent[i]['average_q_values']
                    for i in train_stats_by_agent
                },
                **{
                    f'Train/ByAgent/AverageMCReturns_{i}': train_stats_by_agent[i]['average_mc_returns']
                    for i in train_stats_by_agent
                },
                **{
                    f'Train/ByAgent/Overestimation_{i}': train_stats_by_agent[i]['average_mc_returns'] - train_stats_by_agent[i]['average_q_values']
                    for i in train_stats_by_agent
                },
                'Eval/Active/NumEpisodes': num_episodes_active,
                'Eval/Active/AverageReturns': average_reward_active,
                'Eval/Active/Entropy': entropy_active,
                'Eval/Passive/NumEpisodes': num_episodes_passive,
                'Eval/Passive/AverageReturns': average_reward_passive,
                'Eval/Passive/Entropy': entropy_passive,
                **{
                    f'Eval/ByAgent/AverageReturns_{i}': eval_stats_by_agent[i]['average_returns']
                    for i in eval_stats_by_agent
                },
                **{
                    f'Eval/ByAgent/NumEpisodes_{i}': eval_stats_by_agent[i]['num_episodes']
                    for i in eval_stats_by_agent
                },
                **{
                    f'Eval/ByAgent/AverageQValues_{i}': eval_stats_by_agent[i]['average_q_values']
                    for i in eval_stats_by_agent
                },
                **{
                    f'Eval/ByAgent/AverageMCReturns_{i}': eval_stats_by_agent[i]['average_mc_returns']
                    for i in eval_stats_by_agent
                },
                **{
                    f'Eval/ByAgent/Overestimation_{i}': eval_stats_by_agent[i]['average_mc_returns'] - eval_stats_by_agent[i]['average_q_values']
                    for i in eval_stats_by_agent
                },
            }
            if wandb.run is not None:
                wandb.log(stats, step=self._agent.training_steps)
            if self._has_collector_dispatcher:
                self._collector_dispatcher.write([
                    statistics_instance.StatisticsInstance(
                        'Train/NumEpisodes', num_episodes_train, iteration),
                    statistics_instance.StatisticsInstance(
                        'Train/AverageReturns', average_reward_train,
                        iteration),
                    statistics_instance.StatisticsInstance(
                        'Train/AverageStepsPerSecond',
                        average_steps_per_second, iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Active/NumEpisodes', num_episodes_active,
                        iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Active/AverageReturns', average_reward_active,
                        iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Active/Entropy', entropy_active, iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Passive/NumEpisodes', num_episodes_passive,
                        iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Passive/AverageReturns', average_reward_passive,
                        iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Passive/Entropy', entropy_passive, iteration),
                ])
                self._collector_dispatcher.write([
                    statistics_instance.StatisticsInstance(
                        f'Train/ByAgent/NumEpisodes_{i}',
                        train_stats_by_agent[i]['num_episodes'], iteration)
                    for i in train_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Train/ByAgent/AverageReturns_{i}',
                        train_stats_by_agent[i]['average_returns'], iteration)
                    for i in train_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Train/ByAgent/AverageQValues_{i}',
                        train_stats_by_agent[i]['average_q_values'], iteration)
                    for i in train_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Train/ByAgent/AverageMCReturns_{i}',
                        train_stats_by_agent[i]['average_mc_returns'], iteration)
                    for i in train_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Train/ByAgent/Overestimation_{i}',
                        train_stats_by_agent[i]['average_mc_returns'] - train_stats_by_agent[i]['average_q_values'], iteration)
                    for i in train_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Eval/ByAgent/NumEpisodes_{i}',
                        eval_stats_by_agent[i]['num_episodes'], iteration)
                    for i in eval_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Eval/ByAgent/AverageReturns_{i}',
                        eval_stats_by_agent[i]['average_returns'], iteration)
                    for i in eval_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Eval/ByAgent/AverageQValues_{i}',
                        eval_stats_by_agent[i]['average_q_values'], iteration)
                    for i in eval_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Eval/ByAgent/AverageMCReturns_{i}',
                        eval_stats_by_agent[i]['average_mc_returns'], iteration)
                    for i in eval_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Eval/ByAgent/Overestimation_{i}',
                        eval_stats_by_agent[i]['average_mc_returns'] - eval_stats_by_agent[i]['average_q_values'], iteration)
                    for i in eval_stats_by_agent
                ], collector_allowlist=['tensorboard', 'json'])
        else:
            # Two evaluations, one voting, one random
            self.reset_stats_by_agent()
            self._agent.set_eval_policy(eval_policy='random')
            self._agent.reset_entropy_list()
            num_episodes_eval_random, average_reward_eval_random = self._run_eval_phase(
                statistics)
            entropy_random = np.mean(self._agent.entropy_list)
            eval_stats_by_agent = self.get_stats_by_agent()
            if self._agent.ensemble_size == 1:
                # No need to rerun
                num_episodes_eval_vote = num_episodes_eval_random
                average_reward_eval_vote = average_reward_eval_random
                entropy_vote = entropy_random
            else:
                self._agent.reset_entropy_list()
                self._agent.set_eval_policy(eval_policy='vote')
                num_episodes_eval_vote, average_reward_eval_vote = self._run_eval_phase(
                    statistics)
                entropy_vote = np.mean(self._agent.entropy_list)

            stats = {
                'iteration': iteration,
                'Train/NumEpisodes': num_episodes_train,
                'Train/AverageReturns': average_reward_train,
                'Train/AverageStepsPerSecond': average_steps_per_second,
                **{
                    f'Train/ByAgent/AverageReturns_{i}': train_stats_by_agent[i]['average_returns']
                    for i in train_stats_by_agent
                },
                **{
                    f'Train/ByAgent/NumEpisodes_{i}': train_stats_by_agent[i]['num_episodes']
                    for i in train_stats_by_agent
                },
                **{
                    f'Train/ByAgent/AverageQValues_{i}': train_stats_by_agent[i]['average_q_values']
                    for i in train_stats_by_agent
                },
                **{
                    f'Train/ByAgent/AverageMCReturns_{i}': train_stats_by_agent[i]['average_mc_returns']
                    for i in train_stats_by_agent
                },
                **{
                    f'Train/ByAgent/Overestimation_{i}': train_stats_by_agent[i]['average_mc_returns'] - train_stats_by_agent[i]['average_q_values']
                    for i in train_stats_by_agent
                },
                'Eval/Vote/NumEpisodes': num_episodes_eval_vote,
                'Eval/Vote/AverageReturns': average_reward_eval_vote,
                'Eval/Vote/Entropy': entropy_vote,
                'Eval/Random/NumEpisodes': num_episodes_eval_random,
                'Eval/Random/AverageReturns': average_reward_eval_random,
                'Eval/Random/Entropy': entropy_random,
                **{
                    f'Eval/ByAgent/AverageReturns_{i}': eval_stats_by_agent[i]['average_returns']
                    for i in eval_stats_by_agent
                },
                **{
                    f'Eval/ByAgent/NumEpisodes_{i}': eval_stats_by_agent[i]['num_episodes']
                    for i in eval_stats_by_agent
                },
                **{
                    f'Eval/ByAgent/AverageQValues_{i}': eval_stats_by_agent[i]['average_q_values']
                    for i in eval_stats_by_agent
                },
                **{
                    f'Eval/ByAgent/AverageMCReturns_{i}': eval_stats_by_agent[i]['average_mc_returns']
                    for i in eval_stats_by_agent
                },
                **{
                    f'Eval/ByAgent/Overestimation_{i}': eval_stats_by_agent[i]['average_mc_returns'] - eval_stats_by_agent[i]['average_q_values']
                    for i in eval_stats_by_agent
                },
            }
            if wandb.run is not None:
                wandb.log(stats, step=self._agent.training_steps)
            if self._has_collector_dispatcher:
                self._collector_dispatcher.write([
                    statistics_instance.StatisticsInstance(
                        'Train/NumEpisodes', num_episodes_train, iteration),
                    statistics_instance.StatisticsInstance(
                        'Train/AverageReturns', average_reward_train,
                        iteration),
                    statistics_instance.StatisticsInstance(
                        'Train/AverageStepsPerSecond',
                        average_steps_per_second, iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Vote/NumEpisodes', num_episodes_eval_vote,
                        iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Vote/AverageReturns', average_reward_eval_vote,
                        iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Vote/Entropy', entropy_vote, iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Random/NumEpisodes', num_episodes_eval_random,
                        iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Random/AverageReturns',
                        average_reward_eval_random, iteration),
                    statistics_instance.StatisticsInstance(
                        'Eval/Random/Entropy', entropy_random, iteration),
                ])
                self._collector_dispatcher.write([
                    statistics_instance.StatisticsInstance(
                        f'Train/ByAgent/NumEpisodes_{i}',
                        train_stats_by_agent[i]['num_episodes'], iteration)
                    for i in train_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Train/ByAgent/AverageReturns_{i}',
                        train_stats_by_agent[i]['average_returns'], iteration)
                    for i in train_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Train/ByAgent/AverageQValues_{i}',
                        train_stats_by_agent[i]['average_q_values'], iteration)
                    for i in train_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Train/ByAgent/AverageMCReturns_{i}',
                        train_stats_by_agent[i]['average_mc_returns'], iteration)
                    for i in train_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Train/ByAgent/Overestimation_{i}',
                        train_stats_by_agent[i]['average_mc_returns'] - train_stats_by_agent[i]['average_q_values'], iteration)
                    for i in train_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Eval/ByAgent/NumEpisodes_{i}',
                        eval_stats_by_agent[i]['num_episodes'], iteration)
                    for i in eval_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Eval/ByAgent/AverageReturns_{i}',
                        eval_stats_by_agent[i]['average_returns'], iteration)
                    for i in eval_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Eval/ByAgent/AverageQValues_{i}',
                        eval_stats_by_agent[i]['average_q_values'], iteration)
                    for i in eval_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Eval/ByAgent/AverageMCReturns_{i}',
                        eval_stats_by_agent[i]['average_mc_returns'], iteration)
                    for i in eval_stats_by_agent
                ] + [
                    statistics_instance.StatisticsInstance(
                        f'Eval/ByAgent/Overestimation_{i}',
                        eval_stats_by_agent[i]['average_mc_returns'] - eval_stats_by_agent[i]['average_q_values'], iteration)
                    for i in eval_stats_by_agent
                ], collector_allowlist=['tensorboard', 'json'])
        if hasattr(self._agent, 'update_info') and isinstance(
                self._agent.update_info, dict):
            update_info = {
                k: np.array(v).item()
                for k, v in self._agent.update_info.items()
            }
            self._collector_dispatcher.write(
                [
                    statistics_instance.StatisticsInstance(
                        key, value, iteration)
                    for key, value in update_info.items()
                ],
                collector_allowlist=['json']
            )  # tensorboard already written in agent, console is unnecessary
        # if self._summary_writer is not None:
        # self._save_tensorboard_summaries(iteration, num_episodes_train,
        # average_reward_train,
        # num_episodes_eval,
        # average_reward_eval,
        # average_steps_per_second)
        return statistics.data_lists

    def _initialize_episode(self):
        """Initialization for a new episode.

      Returns:
        action: int, the initial action chosen by the agent.
      """
        initial_observation = self._environment.reset()
        # This signals true episode start, instead of the one after life lost
        # self._agent.true_begin_episode()
        return self._agent.begin_episode(initial_observation)

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        logging.info('Beginning training...')
        if self._start_iteration > self._num_iterations:
            logging.warning('num_iterations (%d) < start_iteration(%d)',
                            self._num_iterations, self._start_iteration)
            return

        # iteration: 1 ... self.num_iterations
        for iteration in range(self._start_iteration, self._num_iterations + 1):
            statistics = self._run_one_iteration(iteration)
            if self._use_legacy_logger:
                self._log_experiment(iteration, statistics)
            if self.save_checkpoint:
                self._checkpoint_experiment(iteration)
            if self._has_collector_dispatcher:
                self._collector_dispatcher.flush()
        if self._summary_writer is not None:
            self._summary_writer.flush()
        if self._has_collector_dispatcher:
            self._collector_dispatcher.close()

        # self._agent.save(self._base_dir, f'agent_{iteration}.pkl')
        # Clean
        # with (Path(self._base_dir) / 'complete.txt').open('w') as f:
            # f.write(str(iteration))
        if self.clean_final_checkpoint:
            shutil.rmtree(self._checkpoint_dir)

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        rewards = []
        terminals = []
        q_values = []

        step_number = 0
        total_reward = 0.

        action, info = self._initialize_episode()
        q_values.append(info.pop('q_value'))
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, is_terminal = self._run_one_step(action)

            total_reward += reward
            step_number += 1

            if self._clip_rewards:
                # Perform reward clipping.
                reward = np.clip(reward, -1, 1)
            # We do MC value estimation using clipped rewards
            rewards.append(reward)
            terminals.append(is_terminal)

            if (self._environment.game_over
                    or step_number == self._max_steps_per_episode):
                # Stop the run loop once we reach the true end of episode.
                break
            elif is_terminal:
                # If we lose a life but the episode is not over, signal an artificial
                # end of episode to the agent.
                self._end_episode(reward, is_terminal)
                action, info = self._agent.begin_episode(observation)
                q_values.append(info.pop('q_value'))
            else:
                action, info = self._agent.step(reward, observation)
                q_values.append(info.pop('q_value'))

        self._end_episode(reward, is_terminal)

        if self.stats_by_agent is not None:
            if self._agent.eval_mode:
                agent_id = self._agent.eval_agent_id
            else:
                agent_id = self._agent.current_agent_id
            assert agent_id is not None
            self.stats_by_agent[agent_id]['total_rewards'] += total_reward
            self.stats_by_agent[agent_id]['num_episodes'] += 1

            # Estimating return
            assert len(rewards) == len(terminals) == len(q_values) == step_number
            assert q_values[0] is not None
            returns = np.zeros(len(rewards))
            next_return = 0.0
            for t in reversed(range(len(rewards))):
                returns[t] = rewards[t] + self._agent.gamma * (1.0 - terminals[t]) * next_return
                next_return = returns[t]

            self.stats_by_agent[agent_id]['q_values'] += np.mean(q_values)
            self.stats_by_agent[agent_id]['mc_returns'] += np.mean(returns)
        return step_number, total_reward

    def reset_stats_by_agent(self):
        self.stats_by_agent = defaultdict(lambda: defaultdict(lambda: 0.0))

    def get_stats_by_agent(self):
        for agent_id in self.stats_by_agent:
            assert 'total_rewards' in self.stats_by_agent[agent_id]
            assert 'num_episodes' in self.stats_by_agent[agent_id]
            assert 'q_values' in self.stats_by_agent[agent_id]
            assert 'mc_returns' in self.stats_by_agent[agent_id]
            self.stats_by_agent[agent_id]['average_returns'] = self.stats_by_agent[agent_id]['total_rewards'] / self.stats_by_agent[agent_id]['num_episodes']
            self.stats_by_agent[agent_id]['average_q_values'] = self.stats_by_agent[agent_id]['q_values'] / self.stats_by_agent[agent_id]['num_episodes']
            self.stats_by_agent[agent_id]['average_mc_returns'] = self.stats_by_agent[agent_id]['mc_returns'] / self.stats_by_agent[agent_id]['num_episodes']
        return_val = self.stats_by_agent
        self.stats_by_agent = None
        return return_val
