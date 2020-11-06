import argparse

import ray
from ray import tune
from utils import MyCallbacks, custom_eval_function
from ray.tune.logger import pretty_print

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
from trading_env_old import TradingEnv, FrameStack

parser = argparse.ArgumentParser()
parser.add_argument('--data_v', type=str, choices=['r12', 'r19'], default='r12',
                    help="r12 have 62days, r19 have 120days.")
parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[600, 800, 600])
parser.add_argument('--lstm', type=bool, default=False)
parser.add_argument('--gamma', type=float, default=0.998)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--train_batch_size', type=int, default=9000)
parser.add_argument('--target_scale', type=float, default=1)
parser.add_argument('--score_scale', type=float, default=1.5)
parser.add_argument('--profit_scale', type=float, default=0)
parser.add_argument('--ap', type=float, default=0.4)
parser.add_argument('--burn_in', type=int, default=3000)
parser.add_argument('--delay_len', type=int, default=30)
parser.add_argument('--target_clip', type=int, default=5)
parser.add_argument('--auto_follow', type=int, default=0)
parser.add_argument('--action_scheme_id', type=int, choices=[3, 15], default=3)
parser.add_argument('--action_repeat', type=int, default=1)
parser.add_argument('--obs_dim', type=int, choices=[26, 38], default=26,
                    help="26 without alive info, 38 with alive info.")
parser.add_argument('--max_ep_len', type=int, default=3000)
parser.add_argument('--lr', type=float, default=4e-5)
parser.add_argument('--entropy_coeff', type=float, default=0)
parser.add_argument("--stop-timesteps", type=int, default=5e8)
parser.add_argument('--exp_name', type=str, default='APPO')
parser.add_argument('--num_stack', type=int, default=1)
parser.add_argument('--num_stack_jump', type=int, default=3)
# parser.add_argument('--alpha', type=float, default=0, help="alpha > 0 enable sppo.")


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    if args.num_stack > 1:
        env = FrameStack
    else:
        env = TradingEnv

    config = {
        "env": env,
        'log_level': 'INFO',
        "env_config": {
            "data_v": args.data_v,
            "obs_dim": args.obs_dim,
            "action_scheme_id": args.action_scheme_id,
            "action_repeat": args.action_repeat,
            "target_scale": args.target_scale,
            "score_scale": args.score_scale,
            "profit_scale": args.profit_scale,
            "action_punish": args.ap,
            "delay_len": args.delay_len,
            "target_clip": args.target_clip,
            "auto_follow": args.auto_follow,
            "burn_in": args.burn_in,
            "max_ep_len": args.max_ep_len,
            "frame_stack": args.num_stack,
            "jump": args.num_stack_jump,
            "model": 'mlp'
        },
        "callbacks": MyCallbacks,

        "num_workers": args.num_workers,

        # Number of GPUs to allocate to the trainer process. Note that not all
        # algorithms can take advantage of trainer GPUs. This can be fractional
        # (e.g., 0.3 GPUs).
        "num_gpus": 1,

        # Discount factor of the MDP.
        "gamma": args.gamma,

        # Number of steps after which the episode is forced to terminate. Defaults
        # to `env.spec.max_episode_steps` (if present) for Gym envs.
        # "horizon": None,
        # Calculate rewards but don't reset the environment when the horizon is
        # hit. This allows value estimation and RNN state to span across logical
        # episodes denoted by horizon. This only has an effect if horizon != inf.
        # "soft_horizon": False,
        # Don't set 'done' at the end of the episode. Note that you still need to
        # set this if soft_horizon=True, unless your env is actually running
        # forever without returning done=True.
        # "no_done_at_end": False,

        "model": {
            "fcnet_hiddens": args.hidden_sizes,
            "use_lstm": args.lstm,
            # # Max seq len for training the LSTM, defaults to 20.
            # "max_seq_len": 20,
            # # Size of the LSTM cell.
            # "lstm_cell_size": 256,
        },
        # "model": {"fcnet_hiddens": args.hidden_sizes},

        # # APPO-specific configs
        # # Whether to use V-trace weighted advantages. If false, PPO GAE advantages
        # # will be used instead.
        # "vtrace": False,
        #
        # # == These two options only apply if vtrace: False ==
        # # Should use a critic as a baseline (otherwise don't use value baseline;
        # # required for using GAE).
        # "use_critic": True,
        # # If true, use the Generalized Advantage Estimator (GAE)
        # # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        # "use_gae": True,
        # # GAE(lambda) parameter
        # "lambda": 0.97,
        #
        # # == PPO surrogate loss options ==
        # "clip_param": 0.3,
        #
        # # == PPO KL Loss options ==
        # "use_kl_loss": False,
        # "kl_coeff": 1.0,
        # "kl_target": 0.01,

        # == IMPALA optimizer params (see documentation in impala.py) ==
        # "rollout_fragment_length": 200,
        "train_batch_size": args.train_batch_size,
        # "min_iter_time_s": 10,
        # # "num_workers": 2,
        # # "num_gpus": 0,
        # "num_data_loader_buffers": 1,
        # "minibatch_buffer_size": 1,
        # "num_sgd_iter": 1,
        # "replay_proportion": 0.0,
        # "replay_buffer_num_slots": 100,
        # "learner_queue_size": 16,
        # "learner_queue_timeout": 300,
        # "max_sample_requests_in_flight_per_worker": 2,
        # "broadcast_interval": 1,
        # "grad_clip": 40.0,
        # "opt_type": "adam",
        "lr": args.lr,
        "lr_schedule": None,
        # "decay": 0.99,
        # "momentum": 0.0,
        # "epsilon": 0.1,
        # "vf_loss_coeff": 0.5,
        # "entropy_coeff": args.entropy_coeff,
        # "entropy_coeff_schedule": None,

        # Evaluation setting
        # Note that evaluation is currently not parallelized

        # Number of parallel workers to use for evaluation. Note that this is set
        # to zero by default, which means evaluation will be run in the trainer
        # process. If you increase this, it will increase the Ray resource usage
        # of the trainer since evaluation workers are created separately from
        # rollout workers.

        # "evaluation_num_workers": 4,
        # # Optional custom eval function.
        # "custom_eval_function": custom_eval_function,
        # # Enable evaluation, once per training iteration.
        # "evaluation_interval": 150,
        # # Run 1 episodes each time evaluation runs.
        # "evaluation_num_episodes": 1,
        # "evaluation_config": {
        #     "explore": False
        # }

        # === Advanced Resource Settings ===
        # Number of CPUs to allocate per worker.
        # "num_cpus_per_worker": 1,
        # Number of GPUs to allocate per worker. This can be fractional. This is
        # usually needed only if your env itself requires a GPU (i.e., it is a
        # GPU-intensive video game), or model inference is unusually expensive.
        # "num_gpus_per_worker": 0,
    }

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
    }

    print(pretty_print(config))
    exp_name = args.exp_name + "-dataV-" + args.data_v + "-num_workers=" + str(args.num_workers)
    exp_name += "-model=" + str(args.hidden_sizes)[1:-1].replace(" ", "") + "-lstm=" + str(args.lstm) + "-batch_size=" + str(args.train_batch_size)
    exp_name += "-obs_dim" + str(args.obs_dim) + "-as" + str(args.action_scheme_id) + "-action_repeat=" + str(args.action_repeat)
    exp_name += "-auto_follow" + str(args.auto_follow) + "-max_ep_len" + str(args.max_ep_len) + "-burn_in" + str(args.burn_in)
    exp_name += "-fs" + str(args.num_stack) + "-jump" + str(args.num_stack_jump)
    exp_name += "-ts" + str(args.target_scale) + "-ss" + str(args.score_scale) + "-ps" + str(args.profit_scale) + "-ap" + str(args.ap)
    exp_name += "-dl" + str(args.delay_len) + "-clip" + str(args.target_clip)
    exp_name += "-gamma" + str(args.gamma) + "-lr" + str(args.lr) + "-entropy" + str(args.entropy_coeff)  # + "-alpha" + str(args.alpha)
    # if args.restore_model:
    #     exp_name += "-restore_model" + str(args.restore_model)

    tune.run("APPO",
             name=exp_name,
             checkpoint_freq=150,
             config=config,
             stop=stop)

    ray.shutdown()
