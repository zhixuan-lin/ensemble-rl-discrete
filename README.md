# Ensemble RL

This is official code for the Atari experiments in the following ICLR 24 paper:

> [The Curse of Diversity in Ensemble-Based Exploration](https://openreview.net/forum?id=M3QXCOTTk4)
>
> *Zhixuan Lin, Pierluca D'Oro, Evgenii Nikishin, Aaron Courville*

The codebase is built upon [dopamine](https://github.com/google/dopamine).

## Dependencies

Create a `conda` environment and activate

```
conda create -n ensemble-rl-discrete python=3.9
conda activate ensemble-rl-discrete
```

Install Jax and flax (GPU). Note this requires CUDA 11.8:

```
pip install "jax[cuda11_pip]==0.4.24" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.8.0
```

Install other requirements:
```
pip install -r requirements.txt
```

Install dopamine

```
pip install --no-deps git+https://github.com/google/dopamine.git@81f695c1525f2774fbaa205cf19d60946b543bc9
```

Install repository:

```
pip install -e .
```

Download Atari ROMs:

```
AutoROM --accept-license
```

## Running Experiments

Logging to your wandb account with `wandb login`.

The default command and configuration for Bootstrapped DQN ($L=0$) is as follows:

```bash
python ensemble_rl/train.py  \
    --exp debug \
    --base_dir './output' \
    --gin_files ensemble_rl/configs/boot_dqn_rmsprop.gin \
    --gin_bindings="atari_lib.create_atari_environment.game_name = 'Pong'" \
    --gin_bindings="JaxBootDQNAgent.ensemble_size = 10" \
    --gin_bindings="JaxBootDQNAgent.share_encoder = False" \
    --gin_bindings="Runner.num_iterations = 200" \
    --seed 0 \
```

Results are saved to `./output`. Wandb visualization is also available under the project name `ensemble-rl`.

The configurations for the experiments in the main paper are as follows:

* Double DQN (Figure 2):

  ```bash
  python ensemble_rl/train.py  \
      --exp debug \
      --base_dir './output' \
      --gin_files ensemble_rl/configs/boot_dqn_rmsprop.gin \
      --gin_bindings="atari_lib.create_atari_environment.game_name = 'Pong'" \
      --gin_bindings="JaxBootDQNAgent.ensemble_size = 1" \
      --gin_bindings="JaxBootDQNAgent.share_encoder = False" \
      --gin_bindings="Runner.num_iterations = 200" \
      --seed 0
  ```

* $90\%$-tandem (Figure 3):

  ```bash
  python ensemble_rl/train.py  \
      --exp debug \
      --base_dir './output' \
      --gin_files ensemble_rl/configs/boot_dqn_rmsprop.gin \
      --gin_bindings="atari_lib.create_atari_environment.game_name = 'Pong'" \
      --gin_bindings="JaxBootDQNAgent.ensemble_size = 2" \
      --gin_bindings="JaxBootDQNAgent.tandem = True" \
  		--gin_bindings="JaxBootDQNAgent.active_prob = 0.9" \
      --gin_bindings="JaxBootDQNAgent.share_encoder = False" \
      --gin_bindings="Runner.num_iterations = 200" \
      --seed 0
  ```

* Adjusting the replay buffer size to `B`, with `B=1000000` as an example (Figure 4):

  ```bash
  B=1000000
  python ensemble_rl/train.py  \
      --exp debug \
      --base_dir './output' \
      --gin_files ensemble_rl/configs/boot_dqn_rmsprop.gin \
      --gin_bindings="atari_lib.create_atari_environment.game_name = 'Pong'" \
      --gin_bindings="JaxBootDQNAgent.ensemble_size = 10" \
      --gin_bindings="JaxBootDQNAgent.share_encoder = False" \
      --gin_bindings="Runner.num_iterations = 200" \
      --gin_bindings="OutOfGraphReplayBuffer.replay_capacity = ${B}"
      --seed 0
  
  ```

* Ensemble size `N`, with `N=5` as an example (Figure 5):

  ```bash
  N=5
  python ensemble_rl/train.py  \
      --exp debug \
      --base_dir './output' \
      --gin_files ensemble_rl/configs/boot_dqn_rmsprop.gin \
      --gin_bindings="atari_lib.create_atari_environment.game_name = 'Pong'" \
      --gin_bindings="JaxBootDQNAgent.ensemble_size = ${N}" \
      --gin_bindings="JaxBootDQNAgent.share_encoder = False" \
      --gin_bindings="Runner.num_iterations = 200" \
      --seed 0
  ```

* Bootstrapped DQN ($L=3$) (Figure 7):

  ```bash
  python ensemble_rl/train.py  \
      --exp debug \
      --base_dir './output' \
      --gin_files ensemble_rl/configs/boot_dqn_rmsprop.gin \
      --gin_bindings="atari_lib.create_atari_environment.game_name = 'Pong'" \
      --gin_bindings="JaxBootDQNAgent.ensemble_size = 10" \
      --gin_bindings="JaxBootDQNAgent.share_encoder = True" \
      --gin_bindings="Runner.num_iterations = 200" \
      --seed 0
  ```

* Bootstrapped DQN ($L=0$) + CERL (Figure 7):

  ```bash
  python ensemble_rl/train.py  \
      --exp debug \
      --base_dir './output' \
      --gin_files ensemble_rl/configs/boot_dqn_rmsprop.gin \
      --gin_bindings="atari_lib.create_atari_environment.game_name = 'Pong'" \
      --gin_bindings="JaxBootDQNAgent.ensemble_size = 10" \
      --gin_bindings="JaxBootDQNAgent.share_encoder = False" \
      --gin_bindings="Runner.num_iterations = 200" \
      --gin_bindings="JaxBootDQNAgent.aux_loss = 'final'" \
      --seed 0
  ```


# Citation

If you find this code useful, please cite the following:

```
@inproceedings{
lin2024the,
title={The Curse of Diversity in Ensemble-Based Exploration},
author={Zhixuan Lin and Pierluca D'Oro and Evgenii Nikishin and Aaron Courville},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=M3QXCOTTk4}
}
```

And the dopamine repo:

```
@article{castro18dopamine,
  author    = {Pablo Samuel Castro and
               Subhodeep Moitra and
               Carles Gelada and
               Saurabh Kumar and
               Marc G. Bellemare},
  title     = {Dopamine: {A} {R}esearch {F}ramework for {D}eep {R}einforcement {L}earning},
  year      = {2018},
  url       = {http://arxiv.org/abs/1812.06110},
  archivePrefix = {arXiv}
}
```

