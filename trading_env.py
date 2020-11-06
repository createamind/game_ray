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

        self.obs_dim = 23
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_dim = 8
        self.action_space = spaces.Discrete(self.action_dim)

        self.max_ep_len = env_config['max_ep_len']
        self.ep_len = 0
        self.no_skip_step_len = 0

    def reset(self):

        start_day = np.random.randint(1, self.trainning_set + 1, 1)[0]  # first self.trainning_set days
        day_index = start_day - 1
        max_point = self.data_len[day_index] - self.max_ep_len - 50
        start_skip = int(np.random.randint(0, max_point, 1)[0])

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

        self.ep_len = 0

        obs = self._get_obs()

        return obs

    def step(self, action):
        self.expso.Action(self.ctx, self.actions[action])
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
        reward = self.rewards[1]/100
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

        if self.obs_dim == 23:
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
        self.jump = env_config['jump']
        self.model = env_config['model']

        self.total_frame = self.frame_stack * self.jump
        self.frames = deque([], maxlen=self.total_frame)
        if self.model == 'mlp':
            self.obs_dim = self.observation_space.shape[0] * self.frame_stack
            self.observation_space = Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        else:
            self.observation_space = Box(-np.inf, np.inf, shape=(self.frame_stack, self.observation_space.shape[0]),
                                         dtype=np.float32)

    def reset(self):
        ob = super().reset()
        ob = np.float32(ob)
        for _ in range(self.total_frame):
            self.frames.append(ob)
        return self.observation()

    def step(self, action):
        ob, reward, done, info = super().step(action)
        ob = np.float32(ob)
        self.frames.append(ob)
        return self.observation(), reward, done, info

    def observation(self):
        assert len(self.frames) == self.total_frame
        obs_stack = np.array(self.frames)
        idx = np.arange(0, self.total_frame, self.jump)
        obs = obs_stack[idx]
        if self.model == 'mlp':
            return np.stack(obs, axis=0).reshape((self.obs_dim,))
        else:
            return obs


if __name__ == "__main__":

    env_config = {
        "data_v": 'r12',
        "obs_dim": 14,
        "action_scheme_id": 15,
        "action_repeat": 1,
        "target_scale": 1,
        "score_scale": 2,
        "profit_scale": 0,
        "action_punish": 0.4,
        "delay_len": 30,
        "target_clip": 5,
        "auto_follow": 0,
        "burn_in": 3000,
        "max_ep_len": 3000,
        "frame_stack": 1,
        "jump": 3,
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
