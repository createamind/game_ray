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
                    default="/home/shuai/ray_results/PPO/PPO_TradingEnv_0_2020-07-31_17-34-23hdewwddn/checkpoint_10000/checkpoint-10000")
parser.add_argument('--num_workers', type=int, default=1)


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

    agent = ppo.PPOTrainer(config=config)
    agent.restore(args.checkpoint)

    min_score = 150

    print("*************** Evaluation start... ***************")
    start_time = time.time()

    for i, worker in enumerate(agent.workers.remote_workers()):
        worker.foreach_env.remote(lambda env: env.eval_set(start_day=51 + i))

    ray.get([w.sample.remote() for w in agent.workers.remote_workers()])

    # Note that the above two statements are the equivalent of:
    metrics = collect_metrics(agent.workers.local_worker(),
                              agent.workers.remote_workers())
    print(pretty_print(metrics))

    total_time = time.time() - start_time
    print("evaluation time: {:.2f}s, {:.2f}min".format(total_time, total_time / 60))
    print("*************** Evaluation end. ***************")

    ray.shutdown()
