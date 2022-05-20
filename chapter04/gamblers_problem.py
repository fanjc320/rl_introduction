#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging

matplotlib.use('Agg')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
# file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
with open('logs.log', 'w'):
    pass

# goal
GOAL = 10

# all states, including state 0 and state 100
STATES = np.arange(GOAL + 1)

# probability of head
HEAD_PROB = 0.4

# 这个问题恰好符合mdp，因为当前状态虽然是有可能有很多别的状态转化而来，但是对后续的影响只和当前状态有关，和
# 它是如何转化而来的无关，即当前有金币5个，那么这5个无论是从10-5输回来的还是从3+2赢过来的，对于后续决策没有影响
# 即当前状态的价值只和当前状态有关，
def figure_4_3():
    # state value
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 10.0 # 这个目标并不是最后赢钱数，只是为了反向传播价值，取别的数也可以

    sweeps_history = []

    # value iteration
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in STATES[1:GOAL]:
            # get possilbe actions for current state
            actions = np.arange(min(state, GOAL - state) + 1) 
            # # 和手里有多少钱无关，这里的state就是手里拥有的钱，如果有钱5，就从state=5开始就好了，state=5投入的钱数最多是goal-5，因为在这道题的需求里可以达到目标了
            # actions = np.arange(1, min(state, GOAL - state) + 1) # action 不应该有0，即啥都不做, 所以最好是下式, +1是因为arange半开
            logger.info("state:%s actions:%s", state, actions)
            action_returns = []
            # np.around(action_returns, 2)
            for action in actions:
                action_returns.append(
                    HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])
            new_value = np.max(action_returns) # 取某个状态的最大价值
            logger.info("action_returns:%s new_value:%s",np.around(action_returns, 2), np.around(new_value, 2))
            state_value[state] = new_value
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
        # if delta < 1e-1: # 会产生不同的结果哦!!!!!
            sweeps_history.append(state_value)
            break

    # compute the optimal policy
    # policy = np.zeros(GOAL + 1)
    # for state in STATES[1:GOAL]:
    #     actions = np.arange(min(state, GOAL - state) + 1) # 可以改成np.arange(1, min(state, GOAL - state) + 1)
    #     action_returns = []
    #     for action in actions:
    #         action_returns.append(
    #             HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])

    #     # round to resemble the figure in the book, see
    #     # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
    #     action_tmp = action_returns[1:] # 既然这里只取1以后的，直接在上面去掉0的内容也是一样的
    #     idx = np.argmax(np.round(action_tmp, 5))
    #     policy[state] = actions[idx + 1]

    # compute the optimal policy
    policy = np.zeros(GOAL)
    for state in STATES[1:GOAL]:
        actions = np.arange(1, min(state, GOAL - state) + 1) # 可以改成np.arange(1, min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(
                HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])
                
        idx = np.argmax(np.round(action_returns, 5))
        policy[state] = actions[idx]
    
    logger.info("policy:%s", policy)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('../images/figure_4_3.png')
    plt.close()


if __name__ == '__main__':
    figure_4_3()



