#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class State:
    def __init__(self):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None
        self.end = None

    # compute the hash value for one state, it's unique
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            # 有时候，无论元素在内存中的分布如何，重要的是要以特定的顺序来访问数组。所以nditer提供了一种顺序参数（order parameter ）的方法来实现这一要求。
            # 默认情况下是order = 'K'， 就是上述的访问方式。另外有：order = 'C'和order = 'F'。不妨理解为：C是按行访问，F是按列访问。
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1 # ??????
        return self.hash_val

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row 水平方向是否有下满三个
        for i in range(BOARD_ROWS):
            # print("self.data:",self.data, " i:", i, " self.data[i,:] ",self.data[i, :])
            results.append(np.sum(self.data[i, :]))
        # check columns 竖直方向是否有下满三个
        for i in range(BOARD_COLS):
            # print("self.data:",self.data, " i:", i, " self.data[:, i] ",self.data[:, i])
            results.append(np.sum(self.data[:, i]))

        # check diagonals 对角线是否下满三个
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
            # print("self.data:",self.data, " i:", i, " trace:", trace, " reverse_trace:", reverse_trace)

        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

#平局
        # whether it's a tie -1或1 全部落满棋盘，总和是9
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def new_next_state(self, i, j, symbol): # 每下一步棋，就会产生一个新的状态，在老状态的基础上更新一步,state的data也会改变
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol # [i,j] ->1/-1
        return new_state

    # print the board
    def print_state(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')

def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i][j] == 0:
                new_state = current_state.new_next_state(i, j, current_symbol)
                new_hash = new_state.hash()
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)
                    if not is_end:# 没结束的情况下，这个状态下仍旧可以继续往下走,之所以在这里用-current_symbol，是因为x和o的数量要一致,当然也可以不用这种方式达到目的
                        get_all_states_impl(new_state, -current_symbol, all_states)
                    else:
                        # print("all_states:", all_states)
                        # print("all_states====================:")
                        # for k, v in all_states.items():
                        #     print(k, v[0].data, v[1])
                        pass


def get_all_states():
    current_symbol = 1
    current_state = State()
    all_states = dict()# hash state is_end
    hs = current_state.hash()
    all_states[hs] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states


# all possible board configurations
all_states = get_all_states()


class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def reset(self):# 每一局结束或新一局开始的时候，清空玩家历史所有状态
        self.p1.reset()
        self.p2.reset()

# 带yield的函数是一个生成器，而不是一个函数了，这个生成器有一个函数就是next函数，next就相当于“下一步”生成哪个数，这一次的next开始的地方是接着上一次的next停止的地方执行的
    def alternate(self): # 交换p1和p2
        while True:
            yield self.p1
            yield self.p2

    # @print_state: if True, print each board during the game 直到end,分出胜负、平局
    def play(self, print_state=False):
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        while True:
# next(iterator[, default])
# Retrieve the next item from the iterator by calling its __next__() method. 
# If default is given, it is returned if the iterator is exhausted, otherwise StopIteration is raised.
            player = next(alternator) # 和yield配对 # p1和p2轮替
            i, j, symbol = player.act()
            next_state_hash = current_state.new_next_state(i, j, symbol).hash()# new出来的state保存在哪个容器了? 早就在all_states中了
            current_state, is_end = all_states[next_state_hash] # all_states 存储了整个下棋过程，包括下完和未下完的棋局,这里从all_states中判断是否结束
            self.p1.set_state(current_state) # 两个玩家共享一个state
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if is_end:
                return current_state.winner


# AI player, 用于训练
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = [] # 本局到现在所有历史状态, 和humanplayer不同
        self.greedy = []
        self.symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)


# self.estimations self.states 都是hash为key的，棋盘上的每一种状态对应一种hash
# 根据all_states 的状态，初始化所有状态对应的value
    def set_symbol(self, symbol):
        self.symbol = symbol
        # for key in a_dict:
        for hash_val in all_states: #5478 个元素, 
            state, is_end = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0 # 赢的状态，所以赢的概率是1
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose 平局和失利
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0 # 输的状态，value是 0
            else:
#                 We set the initial
# values of all the other states to 0.5, representing a guess that we have a 50% chance of winning.
                self.estimations[hash_val] = 0.5

    # update value estimation
    def backup(self):
        states = [state.hash() for state in self.states] # self.states 0-7 :# 所有历史状态的hash集合

        for i in reversed(range(len(states) - 1)):# 从倒数第二个状态开始
            state = states[i]
            next_state = states[i + 1]
            estimate = self.estimations[state] # estimations是hash->value, 这里是当前状态的估计值
            next_estimate = self.estimations[next_state] # 下一状态的value
            greedy = self.greedy[i] # 这里的greedy代表true或false,排除exploration??????
            td_error = greedy * (
                next_estimate - estimate 
            )
            # More precisely, the current value of the earlier state is updated to be closer to
# the value of the later state. This can be done by moving the earlier state’s value a fraction of the way
# toward the value of the later state. If we let s denote the state before the greedy move, and s0 the state
# after the move, then the update to the estimated value of s, denoted V (s), can be written as
# V (s) ← V (s) + α[V (s0 ) − V (s)] ,
# where α is a small positive fraction called the step-size parameter, which influences the rate of learning.
# This update rule is an example of a temporal-difference learning method, so called because its changes
# are based on a difference, V (s0 ) − V (s), between estimates at two different times.
            self.estimations[state] += self.step_size * td_error #本质上是用下一状态的value来更新这一状态的value
        
        # logger.info(self.estimations)

    # choose an action based on the state
    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:# 找到左右还没有落子的地方。
                    next_positions.append([i, j])
                    next_states.append(state.new_next_state(
                        i, j, self.symbol).hash())

        if np.random.rand() < self.epsilon: # explore, 在没有落子的地方随机找一个地方落子
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        for hash_val, pos in zip(next_states, next_positions):# 二者长度一样,有对应关系，即此落子此位置导致了新状态
            hs = self.estimations[hash_val]
            values.append((hs, pos)) # 组合成状态的hash和位置
            # values.append((self.estimations[hash_val], pos))
        # to select one of the actions of equal value at random due to Python's sort is stable
        np.random.shuffle(values)
# 1、max(num, key=lambda x:x[0])语法介绍如下：
# key=lambda 元素: 元素[字段索引]
# print(max(C, key=lambda x: x[0]))  
# x:x[]字母可以随意修改，求最大值方式按照中括号[]里面的维度，[0]按照第一维，[1]按照第二维。
        values.sort(key=lambda x: x[0], reverse=True)# 根据value值倒序，取最大value,
        action = values[0][1] # 取出value最大的位置
        action.append(self.symbol) # 位置[i][j],再加落子1/-1
        return action

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)
        # print("self.estimations:", self.estimations)


# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None #player1 1 player2 -1
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key) # 0-8
        i = data // BOARD_COLS
        j = data % BOARD_COLS
        return i, j, self.symbol


def train(epochs, print_every_n=500):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):# 一个epoch对应一局棋
        winner = judger.play(print_state=True)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()
        player2.backup()
        judger.reset()
    player1.save_policy()
    player2.save_policy()


def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")

def test_reverse():
    test = [1,2,3,4,5]
    le = len(test)
    for i in reversed(range(le - 1)):
    # for i in reversed(range(4)):
        print("i:", i, " i+1:", i+1)

# self.estimations states 都是hash为key的，棋盘上的每一种状态对应一种hash
# self.data是0-8
if __name__ == '__main__':
    #1e5 is a number expressed using scientific notation and it means 10 to the 5th power (the e meaning 'exponent')
    #so 1e5 is equal to 100000, both notations are interchangeably meaning the same.
    train(int(1e2))
    compete(int(1e1))
    play()

    # test_reverse()
    # while True:
    #     print("")

