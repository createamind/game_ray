from typing import Dict
import argparse
import numpy as np

import ray
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
from trading_env import TradingEnv, FrameStack

parser = argparse.ArgumentParser()
parser.add_argument('--data_v', type=str, choices=['r12', 'r19'], default='r19',
                    help="r12 have 62days, r19 have 120days.")
parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[800, 800, 600,500,500,500,500,500])
parser.add_argument('--gamma', type=float, default=0.998)
parser.add_argument('--num_workers', type=int, default=80)
parser.add_argument('--train_batch_size', type=int, default=72000)
parser.add_argument('--target_scale', type=float, default=0)
parser.add_argument('--score_scale', type=float, default=1.5)
parser.add_argument('--profit_scale', type=float, default=0)
parser.add_argument('--ap', type=float, default=0)
parser.add_argument('--burn_in', type=int, default=3000)
parser.add_argument('--delay_len', type=int, default=30)
parser.add_argument('--target_clip', type=int, default=5)
parser.add_argument('--auto_follow', type=int, default=16)
parser.add_argument('--action_scheme_id', type=int, choices=[3, 15], default=15)
parser.add_argument('--action_repeat', type=int, default=1)
parser.add_argument('--obs_dim', type=int, choices=[26, 38], default=38,
                    help="26 without alive info, 38 with alive info.")
parser.add_argument('--max_ep_len', type=int, default=300)
parser.add_argument('--lr', type=float, default=9e-5)
parser.add_argument("--stop-timesteps", type=int, default=5e8)
parser.add_argument('--entropy', type=float, default=0.97, help="alpha > 0， 1.5，3.5.. enable sppo.")
# parser.add_argument('--exp_name', type=str, default='inc_ss')
# parser.add_argument('--num_stack', type=int, default=2)
# parser.add_argument('--num_stack_jump', type=int, default=3)
# parser.add_argument('--alpha', type=float, default=0, help="alpha > 0 enable sppo.")


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        # print("episode {} started".format(episode.episode_id))
        episode.user_data["reward_score"] = []
        episode.user_data["reward_target_bias"] = []
        episode.user_data["reward_ap"] = []
        episode.user_data["ep_target_bias"] = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        if episode.last_info_for() is not None:
            reward_score = episode.last_info_for()['reward_score']
            reward_target_bias = episode.last_info_for()['reward_target_bias']
            reward_ap = episode.last_info_for()['reward_ap']
            ep_target_bias = episode.last_info_for()['target_bias']

            episode.user_data["reward_score"].append(reward_score)
            episode.user_data["reward_target_bias"].append(reward_target_bias)
            episode.user_data["reward_ap"].append(reward_ap)
            episode.user_data["ep_target_bias"].append(ep_target_bias)

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        reward_score = np.sum(episode.user_data["reward_score"])
        reward_target_bias = np.sum(episode.user_data["reward_target_bias"])
        reward_ap = np.sum(episode.user_data["reward_ap"])
        ep_target_bias = np.sum(episode.user_data["ep_target_bias"])

        # print("episode {} ended with length {}, target_bias:{}".format(
        #     episode.episode_id, episode.length, ep_target_bias))

        episode.custom_metrics["reward_score"] = reward_score
        episode.custom_metrics["reward_target_bias"] = reward_target_bias
        episode.custom_metrics["reward_ap"] = reward_ap
        episode.custom_metrics["ep_target_bias"] = ep_target_bias
        episode.custom_metrics["ep_score"] = episode.last_info_for()['score']
        episode.custom_metrics["ep_target_bias_per_step"] = ep_target_bias / episode.length

    # def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
    #                   **kwargs):
    #     print("returned sample batch of size {}".format(samples.count))

    # def on_train_result(self, trainer, result: dict, **kwargs):
    #     print("trainer.train() result: {} -> {} episodes".format(
    #         trainer, result["episodes_this_iter"]))
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True

    # def on_postprocess_trajectory(
    #         self, worker: RolloutWorker, episode: MultiAgentEpisode,
    #         agent_id: str, policy_id: str, policies: Dict[str, Policy],
    #         postprocessed_batch: SampleBatch,
    #         original_batches: Dict[str, SampleBatch], **kwargs):
    #     print("postprocessed {} steps".format(postprocessed_batch.count))
    #     if "num_batches" not in episode.custom_metrics:
    #         episode.custom_metrics["num_batches"] = 0
    #     episode.custom_metrics["num_batches"] += 1


def custom_eval_function(trainer, eval_workers):
    """Example of a custom evaluation function.
    Arguments:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    print("*************** Evaluation start... ***************")
    # We configured 2 eval workers in the training config.
    # worker_1, worker_2 = eval_workers.remote_workers()

    # Set different env settings for each worker. Here we use a fixed config,
    # which also could have been computed in each worker by looking at
    # env_config.worker_index (printed in SimpleCorridor class above).
    for i, worker in enumerate(eval_workers.remote_workers()):
        worker.foreach_env.remote(lambda env: env.eval_set(start_day=51 + i))

    # worker_1.foreach_env.remote(lambda env: env.eval_set(start_day=51))
    # worker_2.foreach_env.remote(lambda env: env.eval_set(start_day=52))

    # Calling .sample() runs exactly one episode per worker due to how the
    # eval workers are configured.
    ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    # You can also put custom values in the metrics dict.
    metrics["foo"] = 1
    print("*************** Evaluation end. ***************")
    return metrics


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    config = {
        "env": TradingEnv,
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
            "max_ep_len": args.max_ep_len
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
        "horizon": None,
        # Calculate rewards but don't reset the environment when the horizon is
        # hit. This allows value estimation and RNN state to span across logical
        # episodes denoted by horizon. This only has an effect if horizon != inf.
        "soft_horizon": False,
        # Don't set 'done' at the end of the episode. Note that you still need to
        # set this if soft_horizon=True, unless your env is actually running
        # forever without returning done=True.
        "no_done_at_end": False,

        # "model": {"use_lstm": True},
        "model": {"fcnet_hiddens": args.hidden_sizes},

        # PPO-specific configs
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE(lambda) parameter.
        "lambda": 0.97,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,

        # Size of batches collected from each worker.
        "rollout_fragment_length": 200,

        # Training batch size, if applicable. Should be >= rollout_fragment_length.
        # Samples batches will be concatenated together to a batch of this size,
        # which is then passed to SGD.
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": args.train_batch_size,

        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 128,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        "lr": args.lr,
        # Learning rate schedule.
        "lr_schedule": None,
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        "vf_share_layers": False,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers: True.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": args.entropy,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # Uses the sync samples optimizer instead of the multi-gpu one. This is
        # usually slower, but you might want to try it if you run into issues with
        # the default optimizer.
        "simple_optimizer": False,
        # Whether to fake GPUs (using CPUs).
        # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
        "_fake_gpus": False,

        # # Evaluation setting
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
        # "evaluation_interval": 15,
        # # Run 1 episodes each time evaluation runs.
        # "evaluation_num_episodes": 1,

        # === Advanced Resource Settings ===
        # Number of CPUs to allocate per worker.
        "num_cpus_per_worker": 1,
        # Number of GPUs to allocate per worker. This can be fractional. This is
        # usually needed only if your env itself requires a GPU (i.e., it is a
        # GPU-intensive video game), or model inference is unusually expensive.
        "num_gpus_per_worker": 0,
    }

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
    }

    tune.run("PPO", config=config, stop=stop)

    ray.shutdown()
