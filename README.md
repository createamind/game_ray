# game_ray


game_ray/

    * rl_game/
    * ppo.py
    * trading_env.py

pip install ray[rllib] requests gputil

CUDA_VISIBLE_DEVICES="0" python ppo.py

tensorboard --logdir=~/ray_results/