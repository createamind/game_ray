from typing import Dict
import numpy as np
import time
import os

import ray
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes, collect_metrics


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        # print("episode {} started".format(episode.episode_id))
        episode.user_data["reward_score"] = []
        episode.user_data["reward_target_bias"] = []
        episode.user_data["reward_ap"] = []
        episode.user_data["ep_target_bias"] = []
        episode.user_data["num_no_action"] = 0

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
            if episode.last_action_for() == 0:
                episode.user_data["num_no_action"] += 1

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
        episode.custom_metrics["ep_num_no_action"] = episode.user_data["num_no_action"]

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


class NewCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        # print("episode {} started".format(episode.episode_id))
        episode.user_data["profit"] = []
        episode.user_data["actions"] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        if episode.last_info_for() is not None:
            profit = episode.last_info_for()['profit']

            episode.user_data["profit"].append(profit)
            episode.user_data["actions"][episode.last_action_for()] += 1

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        profit = np.sum(episode.user_data["profit"])

        # print("episode {} ended with length {}, target_bias:{}".format(
        #     episode.episode_id, episode.length, ep_target_bias))

        episode.custom_metrics["profit"] = profit
        episode.custom_metrics["action_1"] = episode.user_data["actions"][0]
        episode.custom_metrics["action_2"] = episode.user_data["actions"][1]
        episode.custom_metrics["action_3"] = episode.user_data["actions"][2]
        episode.custom_metrics["action_4"] = episode.user_data["actions"][3]
        episode.custom_metrics["action_5"] = episode.user_data["actions"][4]
        episode.custom_metrics["action_6"] = episode.user_data["actions"][5]
        episode.custom_metrics["action_7"] = episode.user_data["actions"][6]


def custom_eval_function(trainer, eval_workers):
    """Example of a custom evaluation function.
    Arguments:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    print("*************** Evaluation start... ***************")
    start_time = time.time()
    # collect_time = time.time()
    metrics = collect_metrics(eval_workers.local_worker(), eval_workers.remote_workers(), timeout_seconds=3)
    # print("collect metrics time:", time.time() - collect_time)
    # print(metrics)

    # save checkpoint here
    if metrics['custom_metrics']:

        ld = os.listdir(trainer.logdir)
        his_score = [float(name.split('_')[1]) for name in ld if name.split('_')[0] == 'score']
        if his_score:
            min_score = min(his_score)
        else:
            min_score = 150
        score = metrics['custom_metrics']['ep_score_mean']
        # if metrics['episode_len_mean'] > 22000 and score < min_score:
        if score < min_score:
            checkpoint_dir = os.path.join(trainer.logdir, "score_{}".format(score))
            checkpoint = trainer.save(checkpoint_dir)
            print("checkpoint saved at", checkpoint)
    else:
        print("no custom_metrics")
    for i, worker in enumerate(eval_workers.remote_workers()):
        worker.foreach_env.remote(lambda env: env.eval_set(start_day=91 + i))

    [w.sample.remote() for w in eval_workers.remote_workers()]

    total_time = time.time() - start_time
    print("evaluation time: {:.2f}s, {:.2f}min".format(total_time, total_time / 60))
    print("*************** Evaluation end. ***************")
    return metrics
