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
from matplotlib.table import Table

matplotlib.use('Agg')

# 书里有两种gridword，这里对应4-1,随机策略，左上角和右下角是终点
WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25


def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


def step(state, action):
    if is_terminal(state):
        return state, 0

    next_state = (np.array(state) + action).tolist()# 不加tolist可以吗
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    reward = -1
    return next_state, reward

#https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.table.html
def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    len_img = len(image)
        # Row and column labels...
    for i in range(len_img):
        # -1 是相对于上面的tb表格，表头自然是-1
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='red', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='green', facecolor='yellow')
    ax.add_table(tb)


def compute_state_value(in_place=True, discount=1.0):
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    while True:
        if in_place:
            state_values = new_state_values # 在markA处产生的价值函数会在这个动作之后，下一个动作之前生效，而不是一轮完成后生效
        else:
            state_values = new_state_values.copy() # 上一轮产生的价值函数
        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                for action in ACTIONS:
                    # 不同的状态通过一些动作，可能对应相同的新状态，奖励也不同，说明奖励不仅和当前状态有关
                    # 有没有可能相同状态通过，不同的动作转到相同的新状态，然后奖励是不一样的，那么奖励不仅和新状态有关，也和动作有关。
                    # 这里step采取固定策略,不会迭代
                    (next_i, next_j), reward = step([i, j], action)#老的状态+动作，返回新的状态,和奖励,这里的状态不需要像棋类那样是整个棋盘
                    # 这里的stat_values是上一个价值函数产生的价值，第一次虽然是0，但是经过第一轮(走完一次终点算一轮)之后，就不是0了
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])# value以动作价值的权重和表示的形式
                new_state_values[i, j] = value #markA 更新价值函数, 这个价值函数会在下一轮生效，而不是立即生效，对于此例，立即生效和一轮以后生效无差别

        max_delta_value = abs(old_state_values - new_state_values).max()
        # 贝尔曼最优解的终止迭代条件就是策略不再能改进，已达到局部最优，往往也是全局最优,是不是对于mdp，局部最优就是全局最优?????
        if max_delta_value < 1e-4:
            break

        iteration += 1

    return new_state_values, iteration


def figure_4_1():
    # While the author suggests using in-place iterative policy evaluation,
    # Figure 4.1 actually uses out-of-place version.
    _, asycn_iteration = compute_state_value(in_place=True)
    values, sync_iteration = compute_state_value(in_place=False)
    roundv = np.round(values, decimals=2)# 保留两位小数点
    draw_image(roundv)
    print('In-place: {} iterations'.format(asycn_iteration))
    print('Synchronous: {} iterations'.format(sync_iteration))

    plt.savefig('../images/figure_4_1.png')
    plt.close()


if __name__ == '__main__':
    figure_4_1()
