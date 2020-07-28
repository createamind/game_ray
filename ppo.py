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

    # We configured 2 eval workers in the training config.
    # worker_1, worker_2 = eval_workers.remote_workers()

    # Set different env settings for each worker. Here we use a fixed config,
    # which also could have been computed in each worker by looking at
    # env_config.worker_index (printed in SimpleCorridor class above).
    for i, worker in enumerate(eval_workers.remote_workers()):
        worker.foreach_env.remote(lambda env: env.eval_set(start_day=51+i))

    # worker_1.foreach_env.remote(lambda env: env.eval_set(start_day=51))
    # worker_2.foreach_env.remote(lambda env: env.eval_set(start_day=52))

    print("Evaluation start...")
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
    return metrics


tune.run(
    "PPO",
    config={
        "env": TradingEnv,
        "callbacks": MyCallbacks,
        "num_workers": 8,
        "num_gpus": 1,
        "gamma": 0.998,
        "lambda": 0.97,
        "lr": 4e-5,
        # "sgd_minibatch_size": 256,
        "train_batch_size": 72000,
        # "clip_param": 0.2,
        # "num_sgd_iter": 20,
        # "rollout_fragment_length": 200,
        # "no_done_at_end": True,
        # "model": {"use_lstm": True},
        "model": {"fcnet_hiddens": [600, 800, 600]},
        "env_config": {"data_v": 'r12'},

        "evaluation_num_workers": 4,
        # Optional custom eval function.
        "custom_eval_function": custom_eval_function,
        # Enable evaluation, once per training iteration.
        "evaluation_interval": 1,
        # Run 1 episodes each time evaluation runs.
        "evaluation_num_episodes": 1,
        })
