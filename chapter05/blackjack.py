#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Nicky van Foreest(vanforeest@gmail.com)                        #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
# file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
with open('logs.log', 'w'):
    pass

# actions: hit or stand
ACTION_HIT = 0 # 取牌
ACTION_STAND = 1  #  "stike" in the book 不取牌，停摆
ACTIONS = [ACTION_HIT, ACTION_STAND]

# policy for player
POLICY_PLAYER = np.zeros(22, dtype=np.int)
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT # 12<=点数<20 取牌
POLICY_PLAYER[20] = ACTION_STAND # 20 和21 停止取牌
POLICY_PLAYER[21] = ACTION_STAND

# function form of target policy of player
def target_policy_player(usable_ace_player, player_sum, dealer_card):
    return POLICY_PLAYER[player_sum]

# function form of behavior policy of player
def behavior_policy_player(usable_ace_player, player_sum, dealer_card): # 取牌或者停摆二分随机
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT

# policy for dealer
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT # 12<=点数<17,继续取牌
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND # 17<=点数<=21,停摆

# get a new card
def get_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# get the value of a card (11 for ace).
def card_value(card_id):
    return 11 if card_id == 1 else card_id

# play a game
# @policy_player: specify policy for player
# @initial_state: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# @initial_action: the initial action
def play(policy_player, initial_state=None, initial_action=None):
    # player status

    # sum of player
    player_sum = 0

    # trajectory of player
    player_trajectory = []

    # whether player uses Ace as 11
    usable_ace_player = False

    # dealer status
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False

    if initial_state is None:
        # generate a random initial state
# 12之前必然hit，所以初始状态里包含了12之前的状态,作为初始状态
        while player_sum < 12: ## markAAAA
            # if sum of player is less than 12, always hit
            card = get_card()
            player_sum += card_value(card)

            # If the player's sum is larger than 21, he may hold one or two aces. 因为只有11+11才能>21,所以应该是两张a
            if player_sum > 21:
                assert player_sum == 22
                # last card must be ace
                player_sum -= 10 # 两张a算12,不要纠结，就是一个初始状态, 用一张或3张也可以
            else:
                usable_ace_player |= (1 == card) # 第一张牌是否是A?

        # initialize cards of dealer, suppose dealer will show the first card he gets
        dealer_card1 = get_card()
        dealer_card2 = get_card()

    else:
        # use specified initial state
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    # initial state of the game 
    state = [usable_ace_player, player_sum, dealer_card1] # exp.[True,16,2]
    # logger.info("play state:%s", state) # state >=12

    # initialize dealer's sum
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2) #  1 in (dealer_card1, dealer_card2)是个bool，然后赋值
    # if the dealer's sum is larger than 21, he must hold two aces.
    if dealer_sum > 21:
        assert dealer_sum == 22
        # use one Ace as 1 rather than 11
        dealer_sum -= 10
    assert dealer_sum <= 21
    assert player_sum <= 21

    # game starts!
# 这里player和dealer是分开执行的，虽然实际玩的时候是轮流的，因为这里不会根据对方是否stand或者hit去改变决策，所以分开执行是ok的
    # player's turn
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            # get action based on current sum
            action = policy_player(usable_ace_player, player_sum, dealer_card1)

        # track player's trajectory for importance sampling
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])
        # logger.info("--play player_sum:%s", player_sum)

        if action == ACTION_STAND:#停止取牌
            break
        # if hit, get new card
        card = get_card()
        # Keep track of the ace count. the usable_ace_player flag is insufficient alone as it cannot
        # distinguish between having one ace or two.
        ace_count = int(usable_ace_player) # usable_ace_player 是否含有被算作11的a，算作1的不计入
        if card == 1:
            ace_count += 1
        player_sum += card_value(card)
        # If the player has a usable ace, use it as 1 to avoid busting and continue.
        while player_sum > 21 and ace_count: # 超过21点，减少一个可用a数量
            player_sum -= 10
            ace_count -= 1
        # player busts
        if player_sum > 21:
            return state, -1, player_trajectory
        assert player_sum <= 21
        usable_ace_player = (ace_count == 1) # ace_count==1来做判断，是因为最多只有1个算作11的a

    # dealer's turn
    while True:
        # get action based on current sum
        action = POLICY_DEALER[dealer_sum]
        if action == ACTION_STAND:
            break
        # if hit, get a new card
        new_card = get_card()
        ace_count = int(usable_ace_dealer)
        if new_card == 1:
            ace_count += 1
        dealer_sum += card_value(new_card)
        # If the dealer has a usable ace, use it as 1 to avoid busting and continue.
        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1
        # dealer busts
        if dealer_sum > 21:
            return state, 1, player_trajectory
        usable_ace_dealer = (ace_count == 1)

    # compare the sum between player and dealer
    assert player_sum <= 21 and dealer_sum <= 21
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory # state:exp.[False,18,10]

# Monte Carlo Sample with On-Policy
def monte_carlo_on_policy(episodes):
    states_usable_ace = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    states_usable_ace_count = np.ones((10, 10)) # playersum:12<=状态初始<=21,见 markAAAA, dealer_card:1-10->10，决策是根据自己的初始状态和对方的第一张明牌进行的
    states_no_usable_ace = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    states_no_usable_ace_count = np.ones((10, 10))
    for i in tqdm(range(0, episodes)):
        _, reward, player_trajectory = play(target_policy_player)
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            # logger.info("=player_sum:%s", player_sum) #最小是12
            player_sum -= 12
            dealer_card -= 1
            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usable_ace[player_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward
    # 一共有2*10*10,200种状态,返回固定策略下每种状态得到的奖励
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count

# 书中的内容:
# The Monte Carlo methods for
# this are essentially the same as just presented for state values, except now we talk about visits to a
# state–action pair rather than to a state.
# The only complication is that many state–action pairs may never be visited
# One way to do this is by specifying that the episodes start in a state–action pair, and that
# every pair has a nonzero probability of being selected as the start.
# . The Monte Carlo ES method
# developed above is an example of an on-policy method.

# Monte Carlo with Exploring Starts
def monte_carlo_es(episodes):
    # (playerSum, dealerCard, usableAce, action)
    state_action_values = np.zeros((10, 10, 2, 2))
    # initialze counts to 1 to avoid division by 0
    state_action_pair_count = np.ones((10, 10, 2, 2))

    # behavior policy is greedy
    def behavior_policy(usable_ace, player_sum, dealer_card):
        usable_ace = int(usable_ace)
        #因为是从玩家的牌面12状态开始算起，所以要先减去12 就是矩阵的0,x位置了
        player_sum -= 12
        dealer_card -= 1
        # get argmax of the average returns(s, a)
        fenzi = state_action_values[player_sum, dealer_card, usable_ace, :]
        fenmu = state_action_pair_count[player_sum, dealer_card, usable_ace, :]
        values_ =  fenzi / fenmu
        # if 是对for遍历 的过滤?只有满足if的才会返回出来, 注意这里action是index，取价值最大的index，可能有1-2个，如果价值相同，就有两个，所以随机一个
        tmp = [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)]
        # logger.info("values_:%s", values_)
        # logger.info("tmp:%s", tmp)
        res = np.random.choice(tmp)
        # logger.info("res:%s", res)

# values_:[ 0.  -0.5]
# tmp:[0]
# res:0
# values_:[-0.5  0. ]
# tmp:[1]
# res:1

        return res

    # play for several episodes
    for episode in tqdm(range(episodes)):
        # for each episode, use a randomly initialized state and action
        # usable_ace_player, player_sum, dealer_card1
        initial_state = [bool(np.random.choice([0, 1])),
                       np.random.choice(range(12, 22)),
                       np.random.choice(range(1, 11))]
        initial_action = np.random.choice(ACTIONS)
        # 书中内容:
        # As the initial policy we use the policy
# evaluated in the previous blackjack example, that which sticks only on 20 or 21.
# 第一次使用的是目标策略（也就是选择最优），后面都是用behavior_policy
        current_policy = behavior_policy if \
                episode else \
                target_policy_player
        _, reward, trajectory = play(current_policy, initial_state, initial_action)
        first_visit_check = set()
        for (usable_ace, player_sum, dealer_card), action in trajectory:
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            state_action = (usable_ace, player_sum, dealer_card, action)
            if state_action in first_visit_check:
                continue
            first_visit_check.add(state_action)
            # update values of state-action pairs
            state_action_values[player_sum, dealer_card, usable_ace, action] += reward
            state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1

    return state_action_values / 9

# Monte Carlo Sample with Off-Policy
def monte_carlo_off_policy(episodes):
    initial_state = [True, 13, 2]

    rhos = []
    returns = []

    for i in range(0, episodes):
        _, reward, player_trajectory = play(behavior_policy_player, initial_state=initial_state)

        # get the importance ratio
        numerator = 1.0
        denominator = 1.0
        for (usable_ace, player_sum, dealer_card), action in player_trajectory:
            if action == target_policy_player(usable_ace, player_sum, dealer_card):
                denominator *= 0.5
            else:
                numerator = 0.0
                break
        rho = numerator / denominator
        rhos.append(rho)
        returns.append(reward)

    rhos = np.asarray(rhos)
    returns = np.asarray(returns)
    weighted_returns = rhos * returns

    weighted_returns = np.add.accumulate(weighted_returns)
    rhos = np.add.accumulate(rhos)

    ordinary_sampling = weighted_returns / np.arange(1, episodes + 1)

    with np.errstate(divide='ignore',invalid='ignore'):
        weighted_sampling = np.where(rhos != 0, weighted_returns / rhos, 0)

    return ordinary_sampling, weighted_sampling

def figure_5_1():
    # states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(1000)
    # states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)
    states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(50000)

    states = [states_usable_ace_1,
              states_usable_ace_2,
              states_no_usable_ace_1,
              states_no_usable_ace_2]

    titles = ['Usable Ace, 10000 Episodes',
              'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes',
              'No Usable Ace, 500000 Episodes']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for state, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('../images/figure_5_1.png')
    plt.close()

def figure_5_2():
    state_action_values = monte_carlo_es(500000)

    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)

    # get the optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)

    images = [action_usable_ace,
              state_value_usable_ace,
              action_no_usable_ace,
              state_value_no_usable_ace]

    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('../images/figure_5_2.png')
    plt.close()

def figure_5_3():
    true_value = -0.27726
    episodes = 10000
    runs = 100
    error_ordinary = np.zeros(episodes)
    error_weighted = np.zeros(episodes)
    for i in tqdm(range(0, runs)):
        ordinary_sampling_, weighted_sampling_ = monte_carlo_off_policy(episodes)
        # get the squared error
        error_ordinary += np.power(ordinary_sampling_ - true_value, 2)
        error_weighted += np.power(weighted_sampling_ - true_value, 2)
    error_ordinary /= runs
    error_weighted /= runs

    plt.plot(np.arange(1, episodes + 1), error_ordinary, color='green', label='Ordinary Importance Sampling')
    plt.plot(np.arange(1, episodes + 1), error_weighted, color='red', label='Weighted Importance Sampling')
    plt.ylim(-0.1, 5)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel(f'Mean square error\n(average over {runs} runs)')
    plt.xscale('log')
    plt.legend()

    plt.savefig('../images/figure_5_3.png')
    plt.close()


if __name__ == '__main__':
    figure_5_1()
    figure_5_2()
    figure_5_3()


# random.binomial(n, p, size=None)¶
# Draw samples from a binomial distribution.

# Samples are drawn from a binomial distribution with specified parameters, 
# n trials and p probability of success where n an integer >= 0 and p is in the interval [0,1]. 
# (n may be input as a float, but it is truncated to an integer in use)


# Moto caluo 应该是可以评估状态和策略都可以的

# 参考:
# https://towardsdatascience.com/monte-carlo-methods-estimate-blackjack-policy-fcc89df7f029
# https://github.com/MJeremy2017/reinforcement-learning-implementation/blob/master/BlackJack/blackjack_mc.py

# using q_learning to blackjack
# https://www.cs.ou.edu/~granville/paper.pdf

# Chapter 5: Monte Carlo Methods
# https://blog.csdn.net/qq_39537898/article/details/112779150


# 二十一点的玩法
# 拥有最高点数的玩家获胜，其点数必须等于或低于21点；超过21点的玩家称为爆牌（Bust）。2点至10点的牌以牌面的点数计算，
# J、Q、K 每张为10点。A可记为1点或11点，而2-10则按牌面点数算，若玩家会因A而爆牌则A可算为1点。
# 当一手牌中的A算为11点时，这手牌便称为“软牌”（soft hand），因为除非玩者再拿另一张牌，否则不会出现爆牌。
# 庄家在取得17点之前必须要牌，因规则不同会有软17点或硬17点才停牌的具体区分。
# 每位玩家的目的是要取得最接近21点数的牌来击败庄家，但同时要避免爆牌。要注意的是，
# 若玩家爆牌在先即为输，就算随后庄家爆牌也是如此。若玩家和庄家拥有同样点数，这样的状态称为“push”，玩家和庄家皆不算输赢
# 牌桌上通常会印有最小和最大的赌注，每一间赌场的每一张牌桌的限额都可能不同。在第一笔筹码下注后，庄家开始发牌，若是从一副或两副牌中发牌，
# 称为“pitch”牌局；较常见的则是从四副牌中发牌。庄家会发给每位玩家和自己两张牌，
# 庄家的两张牌中会有一张是点数朝上的“明牌”，所有玩家皆可看见，另一张则是点数朝下的“暗牌”

# 游戏规则

# 1. 要牌（hit）：

# 玩家需将赌注置于赌桌上。然后，庄家开始发牌,先玩家后庄家交叉发牌，首次发牌每人两张。给每个玩家发的两张牌，牌面朝上；给自己发的两张牌，一张牌面朝上，一张牌面朝下。K、Q、J 和 10 牌都算作 10 点。 A 牌既可算作 1 点也可算作 11 点，由玩家自己决定。A计为11时是”软”牌（如：A、6点数和为软17），A计为1时是”硬”牌（如：A、6、Q总点数为硬17）。其余所有 2 至 9 牌均按其原面值计算。

# 2. 比较大小：

# 如果玩家拿到的前两张牌是一张 A 和一张 10，就拥有黑杰克 (Blackjack)；此时，如果庄家没有黑杰克，玩家就能赢得 1.5 倍的赌金（2 赔 3）。如果庄家是黑杰克，玩家不是，则庄家收走玩家赌注。

# 没有黑杰克的玩家可以继续拿牌，以使总点数尽可能接近但不超过 21 点,谁最接近21 点,谁就赢,如果点数相同,则平,互不输赢；如果超过 21 点，玩家就会“爆”,庄家无需开牌即可收走玩家的赌注。庄家点数超过21点，庄家赔玩家赌注等量赌金。
# 庄家持牌总点数少于16，则必须要牌，直到超过16或是“爆牌”（超过21点），除非庄家拿到的是软16。如果庄家的总点数等于或多于 17 点，则必须停牌。