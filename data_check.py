import ctypes
import json
import os
import pandas as pd
from collections import deque


os.chdir("./rl_game/game/")

soFile = "./game.so"
expso = ctypes.cdll.LoadLibrary(soFile)

all_data = pd.read_feather("/home/shuai/game_ray/r19_1-90days_data.feather")
future_len = 3

BUY_MAP = {-3: 1, -2: 2, -1: 3, 0: 4, 1: 5, 2: 6, 3: 7}
SELL_MAP = {-3: 8, -2: 9, -1: 10, 0: 11, 1: 12, 2: 13, 3: 14}

waiting_len = 3

# actual_target_q = deque([0]*(waiting_len+1), maxlen=waiting_len+1)
# buy_action_q = deque([0]*waiting_len, maxlen=waiting_len)
# sell_action_q = deque([0]*waiting_len, maxlen=waiting_len)


def get_auto_follow_actions(auto_follow):
    target_num = infos[26]
    actual_num = infos[27]
    auto_follow_actions = []
    buy_order_price = 0
    sell_order_price = 0

    global buy_action_q
    global sell_action_q

    num_buy = len(buy_action_q) - buy_action_q.count(0)
    num_sell = len(sell_action_q) - sell_action_q.count(0)

    estimate_num = actual_num + num_buy - num_sell
    if abs(estimate_num - target_num) > auto_follow:
        future_max_price = max(day_data.iloc[step_len + 1:step_len + 1 + future_len]["LastPrice"])
        future_min_price = min(day_data.iloc[step_len + 1:step_len + 1 + future_len]["LastPrice"])
        if target_num > estimate_num:
            price_diff = future_min_price+1 - infos[1]
            if price_diff > 3:
                price_diff = 3
            elif price_diff < -3:
                price_diff = -3
            a = BUY_MAP[price_diff]
            auto_follow_actions = [18, a]
            sell_action_q = deque([0] * waiting_len, maxlen=waiting_len)
            buy_order_price = infos[1] + price_diff
        else:
            price_diff = future_max_price-1 - infos[1]
            if price_diff > 3:
                price_diff = 3
            elif price_diff < -3:
                price_diff = -3
            a = SELL_MAP[price_diff]
            auto_follow_actions = [15, a]
            buy_action_q = deque([0] * waiting_len, maxlen=waiting_len)
            sell_order_price = infos[1] + price_diff

    return auto_follow_actions, buy_order_price, sell_order_price


def get_cancel_actions():

    global buy_action_q
    global sell_action_q

    #                  V
    #        [2, 2, 2, 2]
    #    [57724, 0, 0]
    #        [0, 0, 0]

    cancel_actions = []

    buy_price = [p for p in buy_action_q if p > 0]

    if buy_action_q[0] != 0:
        if buy_action_q[0] == max(buy_price):
            cancel_actions.append(17)
        else:
            cancel_actions.append(16)
        buy_action_q[0] = 0

    sell_price = [s for s in sell_action_q if s > 0]

    if sell_action_q[0] != 0:
        if sell_action_q[0] == min(sell_price):
            cancel_actions.append(19)
        else:
            cancel_actions.append(20)
        sell_action_q[0] = 0

    return cancel_actions


all_score = []
render = False

for start_day in range(1, 91):
    day_data = all_data[all_data.TradingDay == start_day]

    arr_len = 100

    arr = ctypes.c_int * 1
    arr1 = ctypes.c_int * arr_len

    infos = arr1()
    infos_len = arr()
    actions = arr1()
    action_len = arr()
    rewards = arr1()
    rewards_len = arr()

    start_info = {"date_index": "{} - {}".format(start_day, start_day), "skip_steps": 0}
    ctx = expso.CreateContext(json.dumps(start_info).encode())
    expso.GetInfo(ctx, infos, infos_len)
    expso.GetActions(ctx, actions, action_len)
    expso.GetReward(ctx, rewards, rewards_len)

    step_len = 0
    while infos[50] <= 0:
        expso.Step(ctx)
        expso.GetInfo(ctx, infos, infos_len)
        expso.GetReward(ctx, rewards, rewards_len)
        step_len += 1

    no_action_num = 0
    target_bias = 0

    actual_target_q = deque([0] * (waiting_len + 1), maxlen=waiting_len + 1)
    buy_action_q = deque([0] * waiting_len, maxlen=waiting_len)
    sell_action_q = deque([0] * waiting_len, maxlen=waiting_len)

    action_times = 0

    while True:
        env_actions = []

        target_num = infos[26]
        actual_num = infos[27]
        actual_target_q.append(actual_num)

        # check if order make a deal
        if actual_num > actual_target_q[-2]:
            buy_price = [p for p in buy_action_q if p > 0]
            max_price_index = buy_action_q.index(max(buy_price))
            buy_action_q[max_price_index] = 0
        if actual_num < actual_target_q[-2]:
            sell_price = [s for s in sell_action_q if s > 0]
            min_price_index = sell_action_q.index(min(sell_price))
            sell_action_q[min_price_index] = 0

        if render:
            print("-----------------------")
            print("Step_len:", step_len)
            print("AliveAskPriceNUM:", infos[48])
            print("AliveAskVolumeNUM:", infos[49])
            print("AliveAskPrice3:", infos[45])
            print("AliveAskVolume3:", infos[46])
            print("AliveAskPriceSeq3:", infos[47])
            print("AliveAskPrice2:", infos[42])
            print("AliveAskVolume2:", infos[43])
            print("AliveAskPriceSeq2:", infos[44])
            print("AliveAskPrice1:", infos[39])
            print("AliveAskVolume1:", infos[40])
            print("AliveAskPriceSeq1:", infos[41])
            print(".....")
            print("AskPrice3:", infos[12])
            print("AskVolume3:", infos[13])
            print("AskPrice2:", infos[8])
            print("AskVolume2:", infos[9])
            print("AskPrice1:", infos[4])
            print("AskVolume1:", infos[5])
            print("..")
            print("LastPrice:", infos[1])
            print("Volume:", infos[22])
            print("Target_Num:", infos[26])
            print("Actual_Num:", infos[27])
            print("..")
            print("BidPrice1:", infos[2])
            print("BidVolume1:", infos[3])
            print("BidPrice2:", infos[6])
            print("BidVolume2:", infos[7])
            print("BidPrice3:", infos[10])
            print("BidVolume3:", infos[11])
            print(".....")
            print("AliveBidPrice1:", infos[28])
            print("AliveBidVolume1:", infos[29])
            print("AliveBidPriceSeq1:", infos[30])
            print("AliveBidPrice2:", infos[31])
            print("AliveBidVolume2:", infos[32])
            print("AliveBidPriceSeq2:", infos[33])
            print("AliveBidPrice3:", infos[34])
            print("AliveBidVolume3:", infos[35])
            print("AliveBidPriceSeq3:", infos[36])
            print("AliveBidPriceNUM:", infos[37])
            print("AliveBidVolumeNUM:", infos[38])
            print("###")
            print("actual_target_q:", actual_target_q)
            print("buy_action_q:", buy_action_q)
            print("sell_action_q:", sell_action_q)

        cancel_actions = get_cancel_actions()
        auto_follow_actions, buy_order_price, sell_order_price = get_auto_follow_actions(auto_follow=4)

        buy_action_q.append(buy_order_price)
        sell_action_q.append(sell_order_price)

        env_actions += cancel_actions
        env_actions += auto_follow_actions

        if render:
            print("...")
            print("Action:", env_actions)
            print("-----------------------")

        if auto_follow_actions:
            action_times += 1

        for a in env_actions:
            expso.Action(ctx, a)
        expso.Step(ctx)
        expso.GetInfo(ctx, infos, infos_len)
        expso.GetReward(ctx, rewards, rewards_len)
        while infos[0] == -1:
            expso.Step(ctx)
            expso.GetInfo(ctx, infos, infos_len)
            expso.GetReward(ctx, rewards, rewards_len)
            step_len += 1

        step_len += 1

        target_bias += abs(infos[26] - infos[27])

        done = infos[0] == 1

        if done:
            score = rewards[0]
            profit = rewards[1]
            all_score.append(score)
            print(infos[25], "step_len", step_len, "action_times:", action_times, "target_bias:", target_bias/step_len, "profit", profit, "score", score)
            expso.ReleaseContext(ctx)
            break

print("total day:", len(all_score), "average score:", sum(all_score)/len(all_score))

os._exit(8)
