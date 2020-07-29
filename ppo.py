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
        episode.custom_metrics["ep_target_bias_per_step"] = ep_target_bias/episode.length

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
        worker.foreach_env.remote(lambda env: env.eval_set(start_day=51+i))

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


tune.run(
    "PPO",
    config={
        "env": TradingEnv,
        "env_config": {
            "data_v": 'r12',
            "obs_dim": 26,
            "action_scheme_id": 3,
            "target_scale": 1,
            "score_scale": 1.5,
            "profit_scale": 0,
            "action_punish": 0.4,
            "delay_len": 30,
            "target_clip": 5,
            "auto_follow": 0,
            "burn_in": 3000,
            "max_ep_len": 3000
        },
        "callbacks": MyCallbacks,
        "num_workers": 8,
        "num_gpus": 1,
        "gamma": 0.998,
        # "no_done_at_end": True,
        # "model": {"use_lstm": True},
        "model": {"fcnet_hiddens": [600, 800, 600]},

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
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 72000,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 256,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        "lr": 4e-5,
        # Learning rate schedule.
        "lr_schedule": None,
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        "vf_share_layers": False,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers: True.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.2,
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
        # "evaluation_num_workers": 4,
        # # Optional custom eval function.
        # "custom_eval_function": custom_eval_function,
        # # Enable evaluation, once per training iteration.
        # "evaluation_interval": 15,
        # # Run 1 episodes each time evaluation runs.
        # "evaluation_num_episodes": 1,
        })
