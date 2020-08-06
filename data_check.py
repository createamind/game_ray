import ctypes
import json
import os


os.chdir("./rl_game/game/")

soFile = "./game.so"
expso = ctypes.cdll.LoadLibrary(soFile)

all_score = []

for start_day in range(1, 63):

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

    step = 0
    no_action_num = 0
    target_bias = 0
    target_change = 0

    while True:

        target_num = infos[26]
        actual_num = infos[27]

        if abs(actual_num - target_num) > 6:
            if target_num > actual_num:
                expso.Action(ctx, 15) # 买
                expso.Action(ctx, 18)
                # expso.Action(ctx, 6)
                expso.Action(ctx, 5)
                # expso.Action(ctx, 4)
                # expso.Action(ctx, 3)
                # expso.Action(ctx, 2)
                # expso.Action(ctx, 1)
            else:
                expso.Action(ctx, 15)
                expso.Action(ctx, 18)
                # expso.Action(ctx, 9)
                expso.Action(ctx, 10)
                # expso.Action(ctx, 11)
                # expso.Action(ctx, 12)
                # expso.Action(ctx, 13)
                # expso.Action(ctx, 14)
        else:
            no_action_num += 1

        expso.Step(ctx)
        expso.GetInfo(ctx, infos, infos_len)
        expso.GetReward(ctx, rewards, rewards_len)
        step += 1

        now_day = infos[25]
        score = rewards[0]
        profit = rewards[1]
        done = infos[0]
        target_bias += abs(infos[26]-infos[27])

        if infos[26] != target_num:
            target_change += 1

        if done:
            all_score.append(score)
            print(infos[25], "step", step, "action_num:", step-no_action_num, "target_bias_step:", target_bias/step, "profit", profit, "score", score)
            # print(step, end=", ")
            # print(target_change)
            expso.ReleaseContext(ctx)
            break

# print("total day:", len(all_score), "average score:", sum(all_score)/len(all_score))

os._exit(8)
