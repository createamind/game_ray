import ray
import argparse
import copy
import time
import pickle

from ray.tune.utils import merge_dicts
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes, collect_metrics
import os

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str,
                    default="/home/shuai/ray_results/PPO/PPO_TradingEnv_0_2020-08-03_18-28-526c88zg70/checkpoint_6650/checkpoint-6650")
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--start_day', type=int, default=1)
parser.add_argument('--test_days', type=int, default=62)


if __name__ == "__main__":
    args = parser.parse_args()

    config = {}
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(args.checkpoint)
    print(config_dir)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    with open(config_path, "rb") as f:
        config = pickle.load(f)
        print(config)

    config["num_workers"] = args.num_workers
    # Merge with `evaluation_config`.
    evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
    config = merge_dicts(config, evaluation_config)
    config["batch_mode"] = "complete_episodes"
    config["evaluation_config"] = {"explore": False}

    print(config)
    print(pretty_print(config))

    ray.init()

    evaluator = ppo.PPOTrainer(config=config)
    evaluator.restore(args.checkpoint)

    num_workers = args.num_workers
    start_day = args.start_day
    test_days_remain = args.test_days

    print("*************** Evaluation start... ***************")
    start_time = time.time()

    episode_len_mean = []
    # ep_target_bias_mean = []
    ep_target_bias_per_step_mean = []
    ep_score_mean = []
    ep_num_no_action_mean = []
    total_episodes = 0

    while test_days_remain > 0:

        for i, worker in enumerate(evaluator.workers.remote_workers()):
            if i < test_days_remain:
                print("day{} start test.".format(start_day))
                worker.foreach_env.remote(lambda env: env.eval_set(start_day=start_day))
                start_day += 1

        ray.get([worker.sample.remote() for i, worker in enumerate(evaluator.workers.remote_workers()) if i < test_days_remain])

        metrics = collect_metrics(evaluator.workers.local_worker(), evaluator.workers.remote_workers())

        episode_len_mean.append(metrics['episode_len_mean'])
        # ep_target_bias_mean.append(metrics['custom_metrics']['ep_target_bias_mean'])
        ep_target_bias_per_step_mean.append(metrics['custom_metrics']['ep_target_bias_per_step_mean'])
        ep_score_mean.append(metrics['custom_metrics']['ep_score_mean'])
        ep_num_no_action_mean.append(metrics['custom_metrics']['ep_num_no_action_mean'])
        total_episodes += metrics['episodes_this_iter']

        test_days_remain -= num_workers

    score = sum(ep_score_mean)/len(ep_score_mean)
    episode_len_mean = sum(episode_len_mean)/len(episode_len_mean)
    # target_bias_mean = sum(ep_target_bias_mean)/len(ep_target_bias_mean)
    target_bias_per_step_mean = sum(ep_target_bias_per_step_mean) / len(ep_target_bias_per_step_mean)
    num_no_action_mean = sum(ep_num_no_action_mean) / len(ep_num_no_action_mean)
    total_episodes = total_episodes

    test_result = {
        'score': score,
        'episode_len_mean': episode_len_mean,
        # 'target_bias_mean': target_bias_mean,
        'target_bias_per_step_mean': target_bias_per_step_mean,
        'num_action_mean': episode_len_mean-num_no_action_mean,
        'total_episodes': total_episodes,
    }
    print("###################")
    print(pretty_print(test_result))
    print("###################")
    total_time = time.time() - start_time
    print("evaluation time: {:.2f}s, {:.2f}min".format(total_time, total_time / 60))
    print("*************** Evaluation end. ***************")

    ray.shutdown()
