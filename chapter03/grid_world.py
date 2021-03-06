#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# Example 3.6: Gridworld
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

# Matplotlib is a plotting library. It relies on some backend to actually render the plots.
# The default backend is the agg backend. This backend only renders PNGs.
# On Jupyter notebooks the matplotlib backends are special as they are rendered to the browser.
matplotlib.use('Agg')

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# file_handler = logging.FileHandler('logs.log')
# file_handler.setLevel(logging.DEBUG)
# # formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
# # file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# with open('logs.log', 'w'):
#     pass

# Actions that would take the agent off the grid leave its location
# unchanged, but also result in a reward of −1. Other actions result in a reward of 0, except those that
# move the agent out of the special states A and B. From state A, all four actions yield a reward of +10
# and take the agent to A0 . From state B, all actions yield a reward of +5 and take the agent to B0 .

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS=[ '←', '↑', '→', '↓']


ACTION_PROB = 0.25


def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):

        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"

        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')


    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)

def draw_policy(optimal_values):
    fig, ax = plt.subplots()
    ax.set_axis_off()
#     locstr, optional
# The position of the cell with respect to ax. This must be one of the codes.

# bboxBbox, optional
# A bounding box to draw the table into. If this is not None, this overrides loc.
    tb = Table(ax, bbox=[0, 0, 1, 1]) #?????

    nrows, ncols = optimal_values.shape # 5,5
    width, height = 1.0 / ncols, 1.0 / nrows # 0.2,0.2

    # Add cells
    for (i, j), val in np.ndenumerate(optimal_values):# 遍历optimal_values矩阵的索引和值 # ndenumerate 见文档下面注释
        next_vals=[]
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            optimalv = optimal_values[next_state[0],next_state[1]]
            next_vals.append(optimalv) # 上下左右四个方向的格子的值

        max_vals = np.max(next_vals)
        bool_arr = (next_vals == max_vals) #假如最大值在第3,4个， 则bool_arr为[false,false,true,true]
        new_vals = np.where(bool_arr) # 0:array(2,3),所有为true的值的集合
        best_actions=new_vals[0]# 取第一个，array([2,3],)
        val=''
        for ba in best_actions:
            val+=ACTIONS_FIGS[ba] # 取出方向，方便绘图，如果有多个，就组合 "→↓"

        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"

        tb.add_cell(i, j, width, height, text=val,
                loc='center', facecolor='white')

    # Row and column labels...
    l_values = len(optimal_values)
    for i in range(l_values):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                   edgecolor='none', facecolor='none')

    ax.add_table(tb)

# 利用贝尔曼方程不断进行迭代，直至收敛
def figure_3_2():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('../images/figure_3_2.png')
            plt.close()
            break
        value = new_value

# 直接解贝尔曼方程组（每个状态都对应一个贝尔曼方程）
def figure_3_2_linear_system():
    '''
    Here we solve the linear system of equations to find the exact solution.
    We do this by filling the coefficients for each of the states with their respective right side constant.
    '''
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j]  # current state
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            for a in ACTIONS:
                s_, r = step(s, a)
                index_s_ = np.ravel_multi_index(s_, (WORLD_SIZE, WORLD_SIZE))

                A[index_s, index_s_] += ACTION_PROB * DISCOUNT
                b[index_s] -= ACTION_PROB * r

    x = np.linalg.solve(A, b)
    v = x.reshape(WORLD_SIZE, WORLD_SIZE)
    rv = np.round(v, decimals=2)
    draw_image(rv)
    plt.savefig('../images/figure_3_2_linear_system.png')
    plt.close()

# 对应公式3.19，因为这个问题中每个动作 a 只对应一个确定的 s′和 r，因此上式可以简化为：
# v∗ ( s ) = max_a [r + γ v∗ (s′) ]
# 原文链接：https://blog.csdn.net/weixin_42437114/article/details/109432190
def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('../images/figure_3_5.png')
            plt.close()
            draw_policy(new_value)
            plt.savefig('../images/figure_3_5_policy.png')
            plt.close()
            break
        value = new_value


if __name__ == '__main__':
    figure_3_2_linear_system()
    figure_3_2()
    figure_3_5()


# np.ndenumerate
# a = np.array([[1, 2], [3, 4]])
# for index, x in np.ndenumerate(a):
#     print(index, x)
# (0, 0) 1
# (0, 1) 2
# (1, 0) 3
# (1, 1) 4
 
# >>> arr = np.array([[3,6,6],[4,5,1]])
# >>> np.ravel_multi_index(arr, (7,6))
# array([22, 41, 37], dtype=int64)

# 代码分析：
# 该代码默认以“行”优先。在7x6的数组中，取其中（3，6）、（6，5）、（6，1）位置对应的索引数组中的索引值，如下图：

# 由上图可以得到以下规律：
# [3, 4] ——> 3 * 6 + 4 = 22
# [6, 1] ——> 6 * 6 + 5 = 41
# [6, 5] ——> 6 * 6 + 1 = 37
