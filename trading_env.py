import numpy as np
import gym
from gym import spaces
from gym.spaces import Box
import ctypes
import json
import os
import sys
from collections import deque
import pandas as pd
import pickle
import time


data_v19_len = [
    225013, 225015, 225015, 225015, 225015, 225017, 225015, 225015, 225017, 225015, 225015, 225015, 225015, 225015,
    225015, 225015, 225015, 225015, 225015, 225015, 225015, 225015, 225010, 225015, 225015, 135002, 225015, 225015,
    225015, 225015, 225015, 225017, 225015, 225017, 225015, 225017, 225015, 225015, 225015, 225015, 225017, 225015,
    225015, 225015, 225017, 225017, 225016, 225017, 225015, 225013, 225015, 225015, 225017, 225017, 225014, 225017,
    225015, 225013, 225015, 225017, 225015, 225015, 225015, 225017, 225015, 225017, 225017, 225015, 225015, 225015,
    225017, 225017, 225015, 225015, 225017, 225015, 225015, 225017, 225015, 225015, 225014, 225015, 225015, 225015,
    225015, 225015, 225017, 225017, 225015, 225015, 225015, 225015, 225017, 225015, 225017, 225015, 225015, 225015,
    225015, 99005, 225015, 225017, 99009, 225015, 225015, 225009, 225017, 225015, 225015, 225015, 225013, 225013,
    225015, 225015, 225013, 225015, 225015, 225017, 225015, 126016
]  # 120days


class TradingEnv(gym.Env):

    def __init__(self, env_config):
        super(TradingEnv, self).__init__()

        self.data_len = data_v19_len
        self.trainning_set = 90

        rl_game_dir = os.path.dirname(os.path.abspath(__file__)) + "/rl_game/game/"
        os.chdir(rl_game_dir)
        so_file = "./game.so"
        self.expso = ctypes.cdll.LoadLibrary(so_file)
        self.ctx = None

        arr_len = 100
        arr = ctypes.c_int * 1
        arr1 = ctypes.c_int * arr_len
        self.raw_obs = arr1()
        self.raw_obs_len = arr()
        self.rewards = arr1()
        self.rewards_len = arr()
        self.actions = arr1()
        self.action_len = arr()

        self.ori_obs_dim = 23

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.ori_obs_dim,), dtype=np.float32)
        self.action_dim = 7
        self.action_space = spaces.Discrete(self.action_dim)

        self.max_ep_len = env_config['max_ep_len']
        self.ep_len = 0
        self.no_skip_step_len = 0

        self.his_price = deque(maxlen=5)
        self.start_price = None
        self.his_actions = []

    def reset(self, start_day=None):
        if start_day is None:
            start_day = np.random.randint(1, self.trainning_set + 1, 1)[0]  # first self.trainning_set days
            day_index = start_day - 1
            max_point = self.data_len[day_index] - self.max_ep_len - 50
            start_skip = int(np.random.randint(0, max_point, 1)[0])
        else:
            start_skip = 0
        start_info = {"date_index": "{} - {}".format(start_day, start_day), "skip_steps": start_skip}
        # print(start_info)
        if self.ctx:
            self.close_env()
        self.ctx = self.expso.CreateContext(json.dumps(start_info).encode())
        self.expso.GetActions(self.ctx, self.actions, self.action_len)
        self.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)
        while self.raw_obs[50] <= 0:
            self.expso.Step(self.ctx)
            self.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
            self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)
            self.no_skip_step_len += 1

        self.start_price = self.raw_obs[1]
        self.his_actions = []
        self.ep_len = 0

        obs = self._get_obs()

        return obs

    def test_step(self, action):

        self.expso.Action(self.ctx, int(action))
        self.expso.Step(self.ctx)
        self.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)
        while self.raw_obs[0] == -1:
            self.expso.Step(self.ctx)
            self.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
            self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)
            self.no_skip_step_len += 1
        self.no_skip_step_len += 1
        self.ep_len += 1
        obs = self._get_obs()
        reward = 0
        done = self.raw_obs[0] == 1
        info = {
            "TradingDay": self.raw_obs[25],
            "profit": self.rewards[1],
        }

        return obs, reward, done, info

    def step(self, action):
        action = int(action)
        action += 1
        price_diff = {1:-3, 2:-2, 3:-1, 4:0, 5:1, 6:2, 7:3}
        order_price = self.raw_obs[2]+price_diff[action]
        self.his_actions.append((order_price, price_diff[action]))
        self.his_actions = sorted(self.his_actions, key=lambda i: i[0], reverse=True)
        last_target = self.raw_obs[27]

        self.expso.Action(self.ctx, int(action))
        self.expso.Step(self.ctx)
        self.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)
        while self.raw_obs[0] == -1:
            self.expso.Step(self.ctx)
            self.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
            self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)
            self.no_skip_step_len += 1
        self.no_skip_step_len += 1
        self.ep_len += 1

        obs = self._get_obs()

        # profit = self.rewards[1]
        # reward = (profit+self.start_price-self.raw_obs[1])/100

        reward = 0
        num_deal = self.raw_obs[27]-last_target
        for _ in range(num_deal):
            deal = self.his_actions.pop(0)
            reward += -deal[1]+0.1

        done = self.raw_obs[0] == 1 or self.ep_len == self.max_ep_len
        info = {
            "TradingDay": self.raw_obs[25],
            "profit": self.rewards[1],
        }

        return obs, reward, done, info

    def _get_obs(self):

        price_mean = 26440.28
        price_max = 27952.0
        bid_ask_volume_log_mean = 1.97
        bid_ask_volume_log_max = 6.42
        total_volume_mean = 120755.66
        total_volume_max = 321988.0
        # target_abs_mean = 51.018
        target_mean = 2.55
        target_max = 311.0

        price_filter = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 24, 28, 31, 34, 39, 42, 45]
        bid_ask_volume_filter = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 29, 32, 35, 40, 43, 46]
        total_volume_filter = [22]
        target_filter = [26, 27]
        obs = np.array(self.raw_obs[:51], dtype=np.float32)

        obs[price_filter] = (obs[price_filter] - price_mean) / (price_max - price_mean)
        obs[bid_ask_volume_filter] = (np.log(obs[bid_ask_volume_filter]) - bid_ask_volume_log_mean) / (
                bid_ask_volume_log_max - bid_ask_volume_log_mean)
        obs[total_volume_filter] = (obs[total_volume_filter] - total_volume_mean) / (
                total_volume_max - total_volume_mean)
        obs[target_filter] = (obs[target_filter] - target_mean) / (target_max - target_mean)

        if self.ori_obs_dim == 23:

            self.his_price.append(obs[1])
            obs[22] = max(self.his_price)
            obs[23] = min(self.his_price)

            obs = obs[2:25]
        else:
            print(obs.shape)
            assert False, "incorrect obs_dim!"
        obs[obs < -1] = -1
        obs[obs > 1] = 1

        return obs

    def close_env(self):
        self.expso.ReleaseContext(self.ctx)


class FrameStack(TradingEnv):
    def __init__(self, env_config):
        super().__init__(env_config)

        self.frame_stack = env_config['frame_stack']
        self.model = env_config['model']

        self.total_frame = self.frame_stack
        self.frames = deque([], maxlen=self.total_frame)
        if self.model == 'mlp':
            self.obs_dim = self.observation_space.shape[0] * self.frame_stack
            self.observation_space = Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        else:
            self.observation_space = Box(-np.inf, np.inf, shape=(self.frame_stack, self.observation_space.shape[0]),
                                         dtype=np.float32)

    def reset(self, start_day=None):
        ob = super().reset(start_day)
        ob = np.float32(ob)
        for _ in range(self.total_frame):
            self.frames.append(ob)
        return self.observation()

    def step(self, action):
        ob, reward, done, info = super().test_step(action)
        ob = np.float32(ob)
        self.frames.append(ob)
        return self.observation(), reward, done, info

    def observation(self):
        assert len(self.frames) == self.total_frame
        obs_stack = np.array(self.frames)
        idx = np.arange(0, self.total_frame)
        obs = obs_stack[idx]
        if self.model == 'mlp':
            return np.stack(obs, axis=0).reshape((self.obs_dim,))
        else:
            return obs


if __name__ == "__main__":

    env_config = {
        'frame_stack': 3,
        "max_ep_len": 3000,
        "model": 'mlp'
    }

    env = FrameStack(env_config)

    print(env.obs_dim, env.action_space)

    for i in range(1):

        obs = env.reset()
        step = 1
        print(step, obs)
        t0 = time.time()
        price = 0.0
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            step += 1
            print(step, obs, obs.shape)
            if done or step == 100:
                print(step, 'time:', time.time() - t0)
                break
    os._exit(8)
