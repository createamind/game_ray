# game_ray


game_ray/

    * rl_game/
    * ppo.py
    * trading_env.py

pip install "tensorflow-gpu<2.0,>=1.8.0" ray[rllib] requests gputil

CUDA_VISIBLE_DEVICES="0" python ppo.py

tensorboard --logdir=~/ray_results/