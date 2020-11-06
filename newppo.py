import argparse

import ray
from ray import tune
from utils import NewCallbacks, custom_eval_function
from ray.tune.logger import pretty_print

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from trading_env import TradingEnv, FrameStack


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[300, 400, 300])
parser.add_argument('--lstm', type=bool, default=False)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--train_batch_size', type=int, default=500)
parser.add_argument('--action_repeat', type=int, default=1)
parser.add_argument('--max_ep_len', type=int, default=10)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--entropy_coeff', type=float, default=0)
parser.add_argument("--stop-timesteps", type=int, default=5e8)
parser.add_argument('--exp_name', type=str, default='PPO')
parser.add_argument('--num_stack', type=int, default=3)
parser.add_argument('--restore', type=str, default=None, help="restore checkpoint_path")

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    if args.num_stack > 1:
        env = FrameStack
    else:
        env = TradingEnv

    config = {
        "env": env,
        "env_config": {
            "action_repeat": args.action_repeat,
            "max_ep_len": args.max_ep_len,
            "frame_stack": args.num_stack,
            "model": 'mlp'
        },
        "callbacks": NewCallbacks,
        "num_workers": args.num_workers,
        "num_gpus": 1,
        "gamma": args.gamma,

        "model": {
            "fcnet_hiddens": args.hidden_sizes,
            "use_lstm": args.lstm,
            # # Max seq len for training the LSTM, defaults to 20.
            # "max_seq_len": 20,
            # # Size of the LSTM cell.
            # "lstm_cell_size": 256,
        },

        "lambda": 0.95,
        "train_batch_size": args.train_batch_size,
        "sgd_minibatch_size": 50,
        "num_sgd_iter": 10,
        "lr": args.lr,
        "lr_schedule": [[0, args.lr], [150e6, 1e-6]],
        "entropy_coeff": args.entropy_coeff,

        # Evaluation setting
        # Note that evaluation is currently not parallelized

        # Number of parallel workers to use for evaluation. Note that this is set
        # to zero by default, which means evaluation will be run in the trainer
        # process. If you increase this, it will increase the Ray resource usage
        # of the trainer since evaluation workers are created separately from
        # rollout workers.
        # "evaluation_num_workers": 2,
        # # Optional custom eval function.
        # "custom_eval_function": custom_eval_function,
        # # Enable evaluation, once per training iteration.
        # "evaluation_interval": 150,
        # # Run 1 episodes each time evaluation runs.
        # "evaluation_num_episodes": 1,
        # "evaluation_config": {
        #     "explore": False
        # }
    }

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
    }

    print(pretty_print(config))
    exp_name = args.exp_name + "-num_workers=" + str(args.num_workers)
    exp_name += "-model=" + str(args.hidden_sizes)[1:-1].replace(" ", "") + "-lstm=" + str(args.lstm) + "-batch_size=" + str(args.train_batch_size)
    exp_name += "-action_repeat=" + str(args.action_repeat)
    exp_name += "-max_ep_len" + str(args.max_ep_len)
    exp_name += "-fs" + str(args.num_stack)
    exp_name += "-gamma" + str(args.gamma) + "-lr" + str(args.lr) + "-entropy" + str(args.entropy_coeff)  # + "-alpha" + str(args.alpha)
    # if args.restore_model:
    #     exp_name += "-restore_model" + str(args.restore_model)

    checkpoint_path = args.restore
    tune.run("PPO",
             restore=checkpoint_path,
             name=exp_name,
             checkpoint_freq=150,
             config=config,
             stop=stop)

    # ray.shutdown()
