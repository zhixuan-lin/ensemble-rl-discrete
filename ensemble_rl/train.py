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
r"""The entry point for running a Dopamine agent.
"""

from absl import app
from absl import flags
from absl import logging

from dopamine.discrete_domains import run_experiment
from ensemble_rl.run_experiment import EnsembleRunner, create_agent_custom
import tensorflow as tf
import gin
from functools import partial
from pathlib import Path
import gin.config
from datetime import datetime
import shutil
import os
import uuid
import json
from jax.config import config as jax_config

flags.DEFINE_string('exp', 'debug', 'Experiment name')
flags.DEFINE_integer('seed', 42, 'Not actually used')
flags.DEFINE_boolean('resume', True, 'If not resume, old log folder will be first archived')
flags.DEFINE_boolean('disable_jit', False, 'disable jit')
flags.DEFINE_boolean('wandb', True, 'Use wandb')
flags.DEFINE_string('wandb_project', 'ensemble-rl', 'wandb project')
flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')

FLAGS = flags.FLAGS


def init_wandb(project, base_dir):
    os.makedirs(base_dir, exist_ok=True)

    # Check if wandb id is present
    id_path = Path(base_dir) / 'wandb_run_id.txt'
    if id_path.is_file():
        with id_path.open('r') as f:
            run_id = f.read().strip()
            if run_id:
                logging.info(f'Resuming wandb run {run_id}.')
            else:
                run_id = None
    else:
        run_id = None
    import wandb
    from gin import config

    clean_cfg = {}
    # Otherwise _CONFIG will be nested, somehow this doesn't work in wandb
    for (scope, prefix), key_value in config._CONFIG.items():
        # wandb does not like . in config name
        prefix = prefix.replace('.', '-')
        if len(scope) > 0:
            config_key = f'{scope}-{prefix}'
        else:
            config_key = prefix
        clean_cfg[config_key] = key_value

    clean_cfg.update(FLAGS.flag_values_dict())
    # Convert nested config to qualified names;
    wandb.init(project=project, config=clean_cfg, resume='allow', id=run_id)
    with id_path.open('w') as f:
        f.write(wandb.run.id)

    # Save config, but also keeps old config for reference
    config_path = Path(base_dir) / 'config.json'
    if config_path.is_file():
        # Save old config if it exists
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        archive_config_path = Path(base_dir) / f'config_{date_str}_{unique_id}.json'
        config_path.rename(archive_config_path)
    # Save config
    with config_path.open('w') as f:
        json.dump(clean_cfg, f, indent=2)


def archive_log(base_dir):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    archive_dir = os.path.join(base_dir, f'archive_{date_str}_{unique_id}')

    os.makedirs(archive_dir, exist_ok=False)
    file_names = [
        f for f in os.listdir(base_dir) if not f.startswith('archive')
    ]
    for file_name in file_names:
        shutil.move(os.path.join(base_dir, file_name),
                    os.path.join(archive_dir, file_name))


def main(unused_argv):
    """Main method.
    Args:
        unused_argv: Arguments (unused).
    """
    logging.set_verbosity(logging.INFO)
    # tf.compat.v1.disable_v2_behavior()
    if FLAGS.disable_jit:
        jax_config.update('jax_disable_jit', True)

    base_dir = FLAGS.base_dir
    print('Base dir: ', base_dir)
    gin_files = FLAGS.gin_files
    gin_bindings = FLAGS.gin_bindings

    run_experiment.load_gin_configs(gin_files, gin_bindings)

    # Sanity check
    assert len(gin_files) == 1
    # file_name = Path(gin_files[0]).stem
    # agent_name = gin.config._CONFIG[(
        # '', 'ensemble_rl.run_experiment.create_agent_custom')]['agent_name']
    # assert file_name == agent_name, 'File name "{file_name}" does not match agent name "{agent_name}"'

    if not FLAGS.resume:
        archive_log(FLAGS.base_dir)

    if FLAGS.wandb:
        init_wandb(FLAGS.wandb_project, FLAGS.base_dir)

    runner = EnsembleRunner(base_dir=base_dir,
                            create_agent_fn=create_agent_custom)
    runner.run_experiment()


if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)
