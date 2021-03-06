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

info_names = [
    "Done", "LastPrice", "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1", "BidPrice2", "BidVolume2",
    "AskPrice2", "AskVolume2", "BidPrice3", "BidVolume3", "AskPrice3", "AskVolume3", "BidPrice4",
    "BidVolume4", "AskPrice4", "AskVolume4", "BidPrice5", "BidVolume5", "AskPrice5", "AskVolume5", "Volume",
    "HighestPrice", "LowestPrice", "TradingDay", "Target_Num", "Actual_Num", "AliveBidPrice1",
    "AliveBidVolume1", "AliveBidPrice2", "AliveBidVolume2", "AliveBidPrice3", "AliveBidVolume3",
    "AliveAskPrice1", "AliveAskVolume1", "AliveAskPrice2", "AliveAskVolume2", "AliveAskPrice3",
    "AliveAskVolume3", "score", "profit", "total_profit", "baseline_profit", "action", "designed_reward"
]

data_v12_len = [
    225016, 225018, 225018, 225018, 225018, 225017, 225018, 225016, 225014, 225016, 225016, 225018, 225018, 225015,
    225018, 225016, 177490, 225016, 225018, 225016, 225016, 225016, 225018, 225016, 225018, 225018, 225016, 225016,
    225016, 225018, 225018, 225016, 225016, 225018, 225016, 225016, 225018, 225016, 225016, 225015, 225016, 225016,
    225016, 225016, 192623, 225018, 225018, 225016, 225016, 225016, 225016, 225018, 225016, 225018, 225016, 225016,
    225016, 225016, 99006, 225016, 225018, 99010
]  # 62days

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

        self.data_v = env_config['data_v']

        if self.data_v == "r19":
            self.data_len = data_v19_len
            self.trainning_set = 90
        else:
            self.data_len = data_v12_len
            self.trainning_set = 50

        rl_game_dir = os.path.dirname(os.path.abspath(__file__)) + "/rl_game/game/"
        os.chdir(rl_game_dir)
        so_file = "./game.so"
        self.expso = ctypes.cdll.LoadLibrary(so_file)
        arr_len = 100
        arr1 = ctypes.c_int * arr_len
        arr = ctypes.c_int * 1

        self.ctx = None

        self.actions = arr1()
        self.action_len = arr()
        self.raw_obs = arr1()
        self.raw_obs_len = arr()
        self.rewards = arr1()
        self.rewards_len = arr()

        self._actions = self._action_schemes(env_config['action_scheme_id'])
        self.action_repeat = env_config['action_repeat']
        self.auto_follow = env_config['auto_follow']

        self.ori_obs_dim = env_config['obs_dim']
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.ori_obs_dim,), dtype=np.float32)

        self.max_ep_len = env_config['max_ep_len']
        self.ep_len = 0

        self.his_price = deque(maxlen=30)

        # target
        self.target_diff = deque(maxlen=env_config['delay_len'])  # target delay setting
        self.target_clip = env_config['target_clip']
        # reward
        self.target_scale = env_config['target_scale']
        self.score_scale = env_config['score_scale']
        self.profit_scale = env_config['profit_scale']
        assert not (self.score_scale != 0 and self.profit_scale != 0), "score_scale and profit_scale must have one equal to 0"
        self.ap = env_config['action_punish']
        # env reset
        self.burn_in = env_config['burn_in']
        # statistic
        self.act_sta = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                        15: 0, 16: 0}

        self.eval = False
        self.start_day = None

    def _env_skip(self, burn_in):
        for _ in range(burn_in):
            # a = self.policy_069()
            # self.expso.Action(self.ctx, a)
            self.expso.Step(self.ctx)

    def eval_set(self, start_day):
        self.eval = True
        self.start_day = start_day
        self.reset()

    def reset(self):

        self.act_sta = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                        15: 0, 16: 0}

        if self.start_day is not None:  # if test specific day set start_skip=0 and burn_in=0
            start_day = self.start_day
            start_skip = 0
            burn_in = 0
        else:
            start_day = np.random.randint(1, self.trainning_set + 1, 1)[0]  # first self.trainning_set days
            day_index = start_day - 1
            max_point = self.data_len[day_index] - self.max_ep_len - self.burn_in - 50
            start_skip = int(np.random.randint(0, max_point, 1)[0])
            burn_in = self.burn_in

        start_info = {"date_index": "{} - {}".format(start_day, start_day), "skip_steps": start_skip}
        # print(start_info)
        if self.ctx:
            self.close_env()
        self.ctx = self.expso.CreateContext(json.dumps(start_info).encode())
        self.expso.GetActions(self.ctx, self.actions, self.action_len)
        self._env_skip(burn_in)
        self.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)

        self.ep_len = 0

        obs = self._get_obs(self.raw_obs)

        return obs

    def step(self, action):
        reward = 0.0
        for _ in range(self.action_repeat):
            obs, r, done, info = self._step(action)
            reward += r
            if done:
                return obs, reward, done, info
        return obs, reward, done, info

    def _step(self, action):
        last_target = self.raw_obs[26]
        last_bias = self.raw_obs[26] - self.raw_obs[27]
        last_score = self.rewards[0]
        last_profit = self.rewards[1] - self.rewards[3]

        if self.auto_follow is not 0:
            if abs(last_bias) > self.auto_follow:
                if last_bias > 0:
                    action = 5
                else:
                    action = 10

        self._actions(action)
        self.expso.Step(self.ctx)
        self.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)
        self.ep_len += 1

        obs = self._get_obs(self.raw_obs)
        if self.eval:
            done = bool(self.raw_obs[0])
            if done:
                print("Day", self.raw_obs[25], "len:", self.ep_len, "Profit:", self.rewards[1], "Score:", self.rewards[0])
        else:
            done = bool(self.raw_obs[0]) or self.ep_len == self.max_ep_len

        profit = self.rewards[1]
        baseline_profit = self.rewards[3]
        one_step_score = self.rewards[0] - last_score
        one_step_profit = (self.rewards[1] - self.rewards[3] - last_profit) // 100

        reward_score = one_step_score * self.score_scale

        reward_profit = one_step_profit * self.profit_scale

        target_num = self.raw_obs[26]
        actual_num = self.raw_obs[27]

        target_bias = target_num - actual_num

        # self.target_diff是长度为【10】的队列，存放target每次的差值。队列中的target_diff的总和就是当前总容忍度
        # 与上一步的target差值相比，同号且绝对值变小，代表target向实际target靠近，此target变化不应给惩罚延迟
        if not (last_bias * target_bias >= 0 and abs(last_bias) > abs(target_bias)):
            self.target_diff.append(abs(target_num - last_target))
        target_tolerance = sum(self.target_diff)

        reward_target_bias = abs(target_bias)
        # target delay
        reward_target_bias = max(0, reward_target_bias - target_tolerance)
        # target clip
        # target_clip = round(target_now * 0.05)
        reward_target_bias = max(0, reward_target_bias - self.target_clip)
        reward_target_bias *= self.target_scale

        action_penalization = 0 if action == 0 else 1

        designed_reward = -(reward_target_bias + action_penalization * self.ap + reward_score) + reward_profit

        self.act_sta[action] += 1

        info = {"TradingDay": self.raw_obs[25],
                "one_step_score": one_step_score,
                "one_step_profit": one_step_profit,
                "baseline_profit": baseline_profit,
                "score": self.rewards[0],
                "profit": profit,
                "target_bias": abs(target_bias),
                "ap": self.ap,
                "reward_score": -reward_score,
                "reward_profit": reward_profit,
                "reward_target_bias": -reward_target_bias,
                "reward_ap": -action_penalization * self.ap,
                "target_total_tolerance": target_tolerance + self.target_clip,
                }

        if self.ori_obs_dim > 23:
            self.his_price.append(obs[0])
            obs[22] = max(self.his_price)
            obs[23] = min(self.his_price)

        return obs, designed_reward, done, info

    def _get_obs(self, raw_obs):
        if self.data_v == "r19":
            price_mean = 26440.28
            price_max = 27952.0
            bid_ask_volume_log_mean = 1.97
            bid_ask_volume_log_max = 6.42
            total_volume_mean = 120755.66
            total_volume_max = 321988.0
            # target_abs_mean = 51.018
            target_mean = 2.55
            target_max = 311.0
        else:
            price_mean = 26871.05
            price_max = 28540.0
            bid_ask_volume_log_mean = 2.05
            bid_ask_volume_log_max = 6.43
            total_volume_mean = 56871.13
            total_volume_max = 175383.0
            # target_abs_mean = 100.861
            target_mean = 20.69
            target_max = 485.0

        price_filter = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 24, 28, 30, 32, 36, 38, 40]
        bid_ask_volume_filter = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 29, 31, 33, 37, 39, 41]
        total_volume_filter = [22]
        target_filter = [26, 27]
        obs = np.array(raw_obs[:44], dtype=np.float32)

        obs[price_filter] = (obs[price_filter] - price_mean) / (price_max - price_mean)
        obs[bid_ask_volume_filter] = (np.log(obs[bid_ask_volume_filter]) - bid_ask_volume_log_mean) / (
                bid_ask_volume_log_max - bid_ask_volume_log_mean)
        obs[total_volume_filter] = (obs[total_volume_filter] - total_volume_mean) / (
                total_volume_max - total_volume_mean)
        obs[target_filter] = (obs[target_filter] - target_mean) / (target_max - target_mean)

        if self.ori_obs_dim == 38:
            obs = np.delete(obs, [0, 25, 34, 35, 42, 43])
        elif self.ori_obs_dim == 26:
            obs = obs[:28]
            obs = np.delete(obs, [0, 25])
        elif self.ori_obs_dim == 24:
            obs = obs[:25]
            obs = np.delete(obs, [0])
        elif self.ori_obs_dim == 14:
            obs = np.append(obs[2:14], obs[26:28])
        elif self.ori_obs_dim == 7:
            obs = np.append(obs[1:6], obs[26:28])
        elif self.ori_obs_dim == 2:
            obs = obs[26:28]
        else:
            print(obs.shape)
            assert False, "incorrect obs_dim!"
        obs[obs < -1] = -1
        obs[obs > 1] = 1

        return obs

    def _action_schemes(self, action_scheme_id):

        schemes = {}

        def scheme3(action):
            assert 0 <= action <= 2 or action == 5 or action == 10, "action should be 0,1,2"
            if action == 1:
                self.expso.Action(self.ctx, self.actions[18])  # 如果是买动作，卖方向全撤。
                self.expso.Action(self.ctx, self.actions[5])
            elif action == 2:
                self.expso.Action(self.ctx, self.actions[15])  # 如果是卖动作，买方向全撤。
                self.expso.Action(self.ctx, self.actions[10])
            elif action == 0:
                self.expso.Action(self.ctx, self.actions[action])
            # for auto_clip
            elif action == 5:
                self.expso.Action(self.ctx, self.actions[18])
                self.expso.Action(self.ctx, self.actions[5])
            elif action == 10:
                self.expso.Action(self.ctx, self.actions[15])
                self.expso.Action(self.ctx, self.actions[10])

        schemes[3] = scheme3

        # 根据买卖方向进行自动反方向撤单操作
        def scheme15(action):
            assert 0 <= action <= 14, "action should be 0,1,...,14"
            if 1 <= action <= 7:
                self.expso.Action(self.ctx, self.actions[18])  # 如果是买动作，卖方向全撤。
            elif 8 <= action <= 14:
                self.expso.Action(self.ctx, self.actions[15])  # 如果是卖动作，买方向全撤。
            # 执行action
            self.expso.Action(self.ctx, self.actions[action])

        schemes[15] = scheme15

        # 学习全撤单操作
        def scheme17(action):
            assert 0 <= action <= 16, "action should <=16"
            if action <= 14:
                self.expso.Action(self.ctx, self.actions[action])
            elif action == 15:
                self.expso.Action(self.ctx, self.actions[15])
            elif action == 16:
                self.expso.Action(self.ctx, self.actions[18])

        schemes[17] = scheme17

        # 全部操作
        def scheme21(action):
            assert 0 <= action <= 20, "action should be 0,1,...,20"
            self.expso.Action(self.ctx, self.actions[action])

        schemes[21] = scheme21

        # 这里添加新的scheme...
        # def scheme0(action):
        #     pass
        # schemes[0] = scheme0

        self.action_dim = action_scheme_id
        self.action_space = spaces.Discrete(self.action_dim)

        return schemes[action_scheme_id]

    def auto_follow(self):  # actions: 0,6,9
        if self.raw_obs[26] > self.raw_obs[27]:
            action = 6
        elif self.raw_obs[26] < self.raw_obs[27]:
            action = 9
        else:
            action = 0
        return action

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
        idx = np.arange(0, self.total_frame)
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
