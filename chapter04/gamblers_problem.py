#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
GOAL = 100

# 能达到目标的概率是多少呢???如果是永久的玩，没有100的限制，因为概率是0.4，长久下去必破产，
# 如果达到100就算赢，还是有一定概率完成目标的，如果本金<50，赢的概率不会超过50%，
# 不是，从状态价值函数来看，要到大概60多的时候，才会到50%,状态价值函数值随着拥有金币数增加而增加的，但不是线性的
# 价值函数纵坐标0-1其实就是达到目的的概率,总的来讲，这次赌博如果是只按赢钱数目来计算，期望必然是亏钱的，
# 而如果按达到100来和另外的人对赌，则本钱要60多才有超过50%的概率赢
# 如果按到100作为目标，则赌的次数越少越有利，本钱50，则达到目标的概率是40%,
# 本钱40，则最少要赌两次，那么第一次应该是赌10?????赢了到50，再赌一次就有望达到目的，输了最多输10
#
# all states, including state 0 and state 100
STATES = np.arange(GOAL + 1)

# probability of head
HEAD_PROB = 0.4

def figure_4_3():
    # state value
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    # value iteration
    while True:
        delta = 0.0
        for state in STATES[1:GOAL]:
            # get possilbe actions for current state
            actions = np.arange(min(state, GOAL - state) + 1)# 当state==99,最多再下注1,所有动作就是1-min(拥有的金币,goal-拥有的金币)
            action_returns = []
            for action in actions:
                # 这里已经是算的期望了，是动作以后新状态的期望
                action_returns.append(
                    HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])
            new_value = np.max(action_returns)# 求使期望最大的动作价值
            delta += np.abs(state_value[state] - new_value)
            # update state value
            state_value[state] = new_value
        if delta < 1e-9:
            file_handler = logging.FileHandler('state_value.log')
            logger.info("state_value:%s", state_value)
            break

    # compute the optimal policy
    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        # actions = np.arange(min(state, GOAL - state) + 1)
        actions = np.arange(1, min(state, GOAL - state) + 1)# 可以改为最少下注1，不然必输啊，可以加快下进度,
        # 而且有时候0的回报也是最大的之一, 而取0，肯定是无用的...，为啥回报还会是最大的之一????
        action_returns = []
        # 对于gridworld,value值固定后，策略就按四个方向分支格子的状态，找到最大价值的状态前进就可以了，因为只要选择，就可以确定达到那个状态
        # 而这里还要继续计算各个分支的期望,因为就算选择了某个action,达到某个状态也是概率，还有概率到别的状态，所以这里要计算期望
        # 这个action_returns就是动作价值函数,在某个状态下做了某个动作的期望回报,由于期望是一直迭代期望到最后结果的，实际上包含了对未来结果的全部影响
        for action in actions:
            action_returns.append(
                HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])

        # due to tie, can't reproduce the optimal policy in book
        # 因为平局的原因，得不到书中的结果, 是吗?????
        policy[state] = actions[np.argmax(action_returns)]
        # 以下0的期望回报==2的期望回报
        # state:3 action_returns:[0.009225471067568635, 0.008262499101238737, 0.008193534444676069, 0.009225471067568635] policy:0.0 actions:[0 1 2 3]
        logger.info("state:%s action_returns:%s actions:%s", state, action_returns, actions)
        logger.info("state:%s policy:%s", state, policy[state])
        # round to resemble the figure in the book, see
        # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
        # numpy之np.round()取整不得不填的坑https://blog.csdn.net/m0_49475842/article/details/108867547
        # policy[state] = actions[np.argmax(np.round(action_returns[1:], 5))]

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(state_value)
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('../images/figure_4_3.png')
    plt.close()

if __name__ == '__main__':
    figure_4_3()
