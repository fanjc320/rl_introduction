#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
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

class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @step_size: constant step size for updating estimations
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=4, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward
        logger.info("self.q_true:")
        self.q_true = np.array([0,-1,5,0.5])
        logger.info(self.q_true)
        logger.info("self.q_true end")
        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial
        self.q_estimation_fjc = np.copy(self.q_estimation)
        self.all_reward = np.zeros(self.k) #fjc

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            rnd = np.random.choice(self.indices)
            return rnd, rnd

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        # res = 0
        # res_fjc = 0
        # if fjc:
        q_best_fjc = np.max(self.q_estimation_fjc)
        tmp_fjc = np.where(self.q_estimation_fjc == q_best_fjc)[0]
        act_fjc = np.random.choice(tmp_fjc)
        # else:
        q_best = np.max(self.q_estimation)
        tmp = np.where(self.q_estimation == q_best)[0]
        act = np.random.choice(tmp)
        return act, act_fjc

    # take an action, update estimation for this action
    def step(self, action, fjc = False):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.all_reward[action] += reward
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            if fjc: # 更占用计算资源, 从表现看，效果更好???理论上不应该是一样的吗????
                self.q_estimation_fjc[action] = (self.all_reward[action]) / self.action_count[action]
            else:
                # Incremental Implementation
                self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            # self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])

            # # fjc: 推导过程
            # old_estima = self.q_estimation[action]
            # new_estima = old_estima + (reward - self.q_estimation[action]) / self.action_count[action]
            # self.q_estimation[action] = new_estima
            # # =>
            # self.q_estimation[action] = self.q_estimation[action] + (reward - self.q_estimation[action]) / self.action_count[action]
            # # =>
            # self.q_estimation[action] += (reward- self.q_estimation[action])/ self.action_count[action]
            # # => 
            # self.q_estimation[action] += 1.0/self.action_count[action] * (reward- self.q_estimation[action])
            # # =>
            self.q_estimation[action] += self.step_size * (reward- self.q_estimation[action])

        # logger.info("--------------------------------fjc:")
        # logger.info(self.q_estimation)
        # logger.info(self.q_estimation_fjc)
        return reward


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    rewards_fjc = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    best_action_counts_fjc = np.zeros(rewards_fjc.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action,action_fjc = bandit.act()
                reward = bandit.step(action)
                reward_fjc = bandit.step(action_fjc,fjc=True)
                rewards[i, r, t] = reward
                rewards_fjc[i, r, t] = reward_fjc
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1

                if action_fjc == bandit.best_action:
                    best_action_counts_fjc[i, r, t] = 1

    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)

    mean_best_action_counts_fjc = best_action_counts_fjc.mean(axis=1)
    mean_rewards_fjc = rewards_fjc.mean(axis=1)

    return mean_best_action_counts, mean_rewards, mean_best_action_counts_fjc, mean_rewards_fjc


def figure_2_1():
    plt.subplot(2, 1, 1)
    data = np.random.randn(200, 10) + np.random.randn(10)
    plt.violinplot(dataset=data)
    print("dateset")
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.legend()

    plt.subplot(2, 1, 2)
    data = np.random.randn(200, 5)
    plt.violinplot(dataset=data)
    plt.legend()

    plt.savefig('../images/figure_2_1.png')
    plt.close()


def figure_2_2(runs=100, time=1000):
    # epsilons = [0, 0.1, 0.01]
    epsilons = [0.1]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards, best_action_counts_fjc, rewards_fjc = simulate(runs, time, bandits)

    plt.figure(figsize=(20, 20))

    plt.subplot(2, 2, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='$\epsilon = %.02f$' % (eps))
    for eps, rewards in zip(epsilons, rewards_fjc):
        plt.plot(rewards, label='$\_epsilon_fjc = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 2, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='$\epsilon = %.02f$' % (eps))
    for eps, counts in zip(epsilons, best_action_counts_fjc):
        plt.plot(counts, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    #-----------------------------------------------------------
    logger.info("****************************   fjc   *********************************")
    # bandits = [Bandit(epsilon=eps, sample_averages=True,fjc=True) for eps in epsilons]
    # best_action_counts, rewards = simulate(runs, time, bandits)

    plt.subplot(2, 2, 3)
    for eps, rewards in zip(epsilons, rewards_fjc):
        plt.plot(rewards, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps fjc')
    plt.ylabel('average reward fjc')
    plt.legend()

    plt.subplot(2, 2, 4)
    for eps, counts in zip(epsilons, best_action_counts_fjc):
        plt.plot(counts, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps fjc')
    plt.ylabel('% optimal action fjc')
    plt.legend()



    plt.savefig('../images/figure_2_2.png')
    plt.close()


def figure_2_3(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(epsilon=0, initial=5, step_size=0.1))
    bandits.append(Bandit(epsilon=0.1, initial=0, step_size=0.1))
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label='$\epsilon = 0, q = 5$')
    plt.plot(best_action_counts[1], label='$\epsilon = 0.1, q = 0$')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('../images/figure_2_3.png')
    plt.close()


def figure_2_4(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(epsilon=0, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, sample_averages=True))
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label='UCB $c = 2$')
    plt.plot(average_rewards[1], label='epsilon greedy $\epsilon = 0.1$')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('../images/figure_2_4.png')
    plt.close()


def figure_2_5(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=False, true_reward=4))
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = [r'$\alpha = 0.1$, with baseline',
              r'$\alpha = 0.1$, without baseline',
              r'$\alpha = 0.4$, with baseline',
              r'$\alpha = 0.4$, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.savefig('../images/figure_2_5.png')
    plt.close()


def figure_2_6(runs=2000, time=1000):
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),
                  lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True),
                  lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_averages=True),
                  lambda initial: Bandit(epsilon=0, initial=initial, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(runs, time, bandits)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i+l], label=label)
        i += l
    plt.xlabel('Parameter($2^x$)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('../images/figure_2_6.png')
    plt.close()


if __name__ == '__main__':
    # figure_2_1()
    figure_2_2()
    # figure_2_3()
    # figure_2_4()
    # figure_2_5()
    # figure_2_6()
