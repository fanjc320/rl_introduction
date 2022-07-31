#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

matplotlib.use('Agg')

# https://medium.com/@jaems33/this-is-start-of-my-exploration-into-learning-about-reinforcement-learning-d505a68a2d6
# Policy Iteration

# Here are key specific points about the problem:
# Each location can only hold 20 cars.
# Every time a car is rented, we earn $10 (Reward)
# Every time we move a car overnight to another location, it costs us $2 (Negative Reward).
# The maximum number of cars we can move overnight is 5 (Action).
# The number of cars requested and returned at each location (n) on any given day are Poisson random variables.
# The expected number (lambda) of rental requests at the first and second location is 3 and 4 respectively.
# The expected number of rental returns at the first and second location is 3 and 2 respectively.
# Thus, the second location has more rentals than returns, whereas the first location has an equal number of rentals as returns.
# The time step are days (thus, one step in an iteration can be considered a full day), the state is the number of cars at each location at the end of the day, and the actions are the net number of cars moved between the two locations overnight.


# maximum # of cars in each location
MAX_CARS = 20

# maximum # of cars to move during night
MAX_MOVE_OF_CARS = 5

# expectation for rental requests in first location
RENTAL_REQUEST_FIRST_LOC = 3

# expectation for rental requests in second location
RENTAL_REQUEST_SECOND_LOC = 4

# expectation for # of cars returned in first location
RETURNS_FIRST_LOC = 3

# expectation for # of cars returned in second location
RETURNS_SECOND_LOC = 2

DISCOUNT = 0.9

# credit earned by a car
RENTAL_CREDIT = 10

# cost of moving a car
MOVE_CAR_COST = 2

# all possible actions
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)

# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
POISSON_UPPER_BOUND = 11

# Probability for poisson distribution
# @lam: lambda should be less than 10 for this function
poisson_cache = dict()


def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        # pmf(k, mu, loc=0)
        # Probability mass function.
        # 泊松分布的参数λ是随机事件发生次数的数学期望值
        poisson_cache[key] = poisson.pmf(n, lam)
    
    # print("poisson_probability:", poisson_cache)
    return poisson_cache[key]


def expected_return(state, action, state_value, constant_returned_cars):
    # print("action:", state,action, state_value)
    """
    @state: [# of cars in first location, # of cars in second location]
    @action: positive if moving cars from first location to second location,
            negative if moving cars from second location to first location
    @stateValue: state value matrix
    @constant_returned_cars:  if set True, model is simplified such that
    the # of cars returned in daytime becomes constant
    rather than a random value from poisson distribution, which will reduce calculation time
    and leave the optimal policy/value state matrix almost the same
    """
    # initailize total return
    returns = 0.0

    # cost for moving cars
    returns -= MOVE_CAR_COST * abs(action) # 扣除转移车辆产生的成本
# 本逻辑里有车辆的出租和返还，有收租和迁移成本，构成矛盾
    # moving cars
    NUM_OF_CARS_FIRST_LOC = min(state[0] - action, MAX_CARS)# 现有车辆=location1车的数量-转移的数量,action可正可负,
    NUM_OF_CARS_SECOND_LOC = min(state[1] + action, MAX_CARS)# location2车的数量+转移的数量,action可正可负

    # go through all possible rental requests 租车需求看做泊松分布，最多10个,这个书里并没有这个限制好像
    for rental_request_first_loc in range(POISSON_UPPER_BOUND):# location1租车需求数量0-10,
        for rental_request_second_loc in range(POISSON_UPPER_BOUND):
            # probability for current combination of rental requests
            prob = poisson_probability(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) * \
                poisson_probability(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

            num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOC # location1 action(转移车辆) 后现有车的数量
            num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC # location2 action 后现有车的数量

            # valid rental requests should be less than actual # of cars
            valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)# 有效出租数量=min(现有车的数量，出租的需求数量)
            valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

            # get credits for renting
            reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT # 租车获得的reward
            num_of_cars_first_loc -= valid_rental_first_loc# 租出去后，车剩余的数量
            num_of_cars_second_loc -= valid_rental_second_loc

            if constant_returned_cars:
                # get returned cars, those cars can be used for renting tomorrow
                returned_cars_first_loc = RETURNS_FIRST_LOC
                returned_cars_second_loc = RETURNS_SECOND_LOC
                num_of_cars_first_loc = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS) # 车返还后，车剩余的数量,因为是白天出租，第二天返还，所以一定要在出租后计算
                num_of_cars_second_loc = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                # 这里的return又会作为返回值更新state_value,这里是value或者说return可以不断迭代逼近value的期望值的原因,每次进步一点点,因为state_value一直在进步
                # 逼近期望值, 这里的state_value，是动作做完，也就是下一状态的估计值，reward是真实值
                # ????????
                returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])
            else:
                for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                    for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(
                            returned_cars_first_loc, RETURNS_FIRST_LOC) * poisson_probability(returned_cars_second_loc, RETURNS_SECOND_LOC)
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                        prob_ = prob_return * prob
                        returns += prob_ * (reward + DISCOUNT *
                                            state_value[num_of_cars_first_loc_, num_of_cars_second_loc_])
    return returns


def figure_4_2(constant_returned_cars=True):
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros(value.shape, dtype=np.int)

    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    # 注意这里的求解值限于当天和第二天，第二天的车辆变化，就是另外一个状态而已，所以每天都是一个已被包含的新状态，计算时不需考虑n多天的回报
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        # policy evaluation (in-place)????inplace是指value inplace吗? 此时policy一直是0，为了拿到value?????
        # 这里一共有20*20=400种状态，每种状态的value也是单独统计的
        while True:
            old_value = value.copy()
            # 策略固定下，正是状态的不同,导致了不同状态的价值是不同的，
            # 所以是先固定策略，先根据状态分布来得到不同状态的价值
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    # policy 是不同的动作，索引是location1和location2的车子数量，即状态,值是从location1搬移到location2的车子数量
                    # 这里先求状态函数，policy固定是0，即一辆车都不移动
                    # value是状态价值函数, [i,j]分别是location1和location2车子的数量
                    # 其实policy固定，value的期望值可看做是固定的，这里的循环只是为了不断接近value的期望值???
                    new_state_value = expected_return([i, j], policy[i, j], value, constant_returned_cars)# 函数内部不会改变policy和value
                    value[i, j] = new_state_value# 状态价值在每个action之后变
            max_value_change = abs(old_value - value).max()
            print('max value change {}'.format(max_value_change))
            # if max_value_change < 1e-4:# 新老价值一定会小于这个数而不会死循环吗?无限逼近还是震荡逼近极值？
            if max_value_change < 1e-1: # 因为是mdp问题，状态有限，所以状态价值必会趋于稳定?
                break
        
        # policy improvement, 此时value不再改变，policy改变，根据value，通过贪心找到最优policy
        policy_stable = True
        # 在所有状态下:尝试了所有不同操作,以此得到所有动作，并根据价值期望获得最大期望值，作为最优策略的一部分,最后得到最优策略.
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = policy[i, j]
                action_returns = []
                for action in actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(expected_return([i, j], action, value, constant_returned_cars))# 动作价值
                    else:
                        action_returns.append(-np.inf)
                new_action = actions[np.argmax(action_returns)]# [让价值返回最大的action的索引],取动作价值最大的动作
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:#????? value固定，基本上贪心算法也就固定了，就一种或几种,所以policy是会趋于稳定的
                    policy_stable = False
        print('policy stable {}'.format(policy_stable))
        # 以上两个循环跳出的标志都是趋于稳定，对于价值来讲，就是改变的非常小了，对于策略来讲，就是不再改变，
        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            break

        iterations += 1

    plt.savefig('../images/figure_4_2.png')
    plt.close()


if __name__ == '__main__':
    figure_4_2()
    
    # 每次输出都是一样的，说明上面引用poisson分布也是用了期望值?
    # for i in range(20):
    #     prob = poisson_probability(10, 4)
    #     print("prob:",prob)