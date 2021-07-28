
import gym
import random
import math
import numpy as np
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
import numpy.random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.distributions import Categorical

import time

time_start = time.time()
ENV_NAME = 'CartPole-v0'

plt.ion()

# # 利用gpu加速神经网络计算时会用到，如使用gpu后面一些地方还需要加上to(device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_HIDDEN = 256  # 神经网络隐藏层神经元个数
SEED = 8  # 随机数种子，保证结果可复现


# 定义策略网络，actor和critic共用一个网络主体。网络的深度、宽度、激活函数或许都有更优解
# 函数应该比较简单，所以只用了3层的前馈神经网络（之前参考pytorch官方代码试过只用2层（无隐藏层））
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(4, N_HIDDEN)  # 输入层
        self.layer2 = nn.Linear(N_HIDDEN, N_HIDDEN)  # 隐藏层

        # 2个分支的输出层
        # actor
        self.action_head = nn.Linear(N_HIDDEN, 2)

        # critic
        self.value_head = nn.Linear(N_HIDDEN, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))  # 激活函数可以从relu、sigmoid、tanh等中选取
        action_prob = f.softmax(self.action_head(x), dim=-1)  # 通过softmax函数将权重转化为0~1的概率
        state_value = self.value_head(x)  # 取值为全体实数，或许可以用torch.tanh()映射到（-1，1）
        return action_prob, state_value


class Agent(object):
    def __init__(self):
        self.gamma = 0.9  # 折扣因子
        self.net = Policy()  # 实例化网络
        self.optim1 = optim.Adam(self.net.parameters(), lr=0.0005)  # 设置优化器（随机梯度下降（SGD）算法也可）
        self.log_probs = []  # 存储输出概率的对数（softmax层的输入，也就是self.layer2的输出）
        self.values = []  # 存储预测价值
        self.rewards = []  # 存储一个路径上的全部reward

    def select_action(self, ob):
        ob = torch.Tensor(ob)  # 转化为tensor以便输入网络进行计算
        probs, v = self.net(ob)  # 输入网络进行计算
        m = Categorical(probs)  # 分类器
        a = m.sample()  # 利用分类器依据概率选取动作
        self.log_probs.append(m.log_prob(a))  # 存储输出概率的对数（softmax层的输入，也就是self.layer2的输出）
        self.values.append(v)  # 存储预测价值
        return a.item()  # 返回选择的动作

    def learn(self):

        R = 0
        returns = []  # 存储一整条路径上到的回报
        policy_losses = []  # 存储actor损失
        value_losses = []  # 存储critic损失

        EPS = 0.001  # 防止除以0

        # 获得一整条路径的奖励后，根据蒙特卡洛法计算整条路径的回报
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + EPS)  # 回报数据标准化（这一步是否必要还需进一步讨论）

        for log_prob, value, Re in zip(self.log_probs, self.values, returns):
            # 计算优势函数（Q - V）
            advantage = Re - value.item()  # 因为转化为了list所以这部分自动不计算梯度了
            # 计算loss
            policy_losses.append(-log_prob * advantage)  # log_prob一直是tensor类型，是需要计算梯度的；advantage不需要计算梯度
            value_losses.append(f.smooth_l1_loss(value, torch.tensor([R])))  # value需要计算梯度；R不需要

        # 优化神经网络模型参数
        self.optim1.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()  # 两个损失作和作为总损失
        loss.backward()
        self.optim1.step()

        # 清空缓存数据
        self.rewards = []
        self.log_probs = []
        self.values = []


# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


env = gym.make(ENV_NAME).unwrapped
setup_seed(SEED)
agent = Agent()  # 实例化智能体
average_score = 0
average_scores = []  # 存储平均分

for i_episode in count():

    observation = env.reset()
    episode_score = 0

    for t in count():
        action = agent.select_action(observation)
        observation_, reward, done, _ = env.step(int(action))

        """
        自行设计reward。用gym自带的reward应该也可。前者代表倾倒得到的奖励，后者代表每坚持1回合获得的奖励
        （SEED默认选取8，多次实验则用的是9、10、11等。可根据种子复现结果。时间与所述有差异说明电脑配置有差异）
        目前尝试过的组合：无隐藏层       （-1，0.1）较优（49秒）     ；（0，0.1）较优（55秒）           ；（0，1）较优（44秒）；（-1，0）较差（118秒）
                      1层隐藏层（当前）（-1，0.1）较优（31秒、55秒）；（0，0.1）极佳（29秒、28秒，45秒）；（0，1）较优（59秒）；（-1，0）极佳（28秒、27秒、38秒、28秒）
        """
        reward = -1 if done else 0

        agent.rewards.append(reward)  # 存储整条路径上的reward
        episode_score += 1  # 每坚持一个回合得1分
        # env.render()  # 环境渲染。加了这句可以看到实际情况但速度会严重变慢
        observation = observation_
        # 设置单幕迭代上限。算法能力有限，迭代上限过高可能导致永远达不到标准
        if done or t >= 199:
            if not done:
                env.reset()
            break
    # 计算平均得分（以往得分权重为0.95，当次得分权重为0.05）
    average_score = 0.05 * episode_score + 0.95 * average_score
    average_scores.append(average_score)

    agent.learn()

    # 显示结果
    if i_episode % 10 == 0:
        print('Episode {}\tLast score: {:.2f}\tAverage score: {:.2f}'.format(
            i_episode, episode_score, average_score))

    if average_score > 195:
        time_end = time.time()
        time_c = time_end - time_start
        print("Solved! Average score is now {} , cost {:.2f}s and "
              "the total episode is {} steps!".format(average_score, time_c, i_episode))
        break

# 绘图
plt.title('actor_critic')
plt.xlabel('Episode')
plt.ylabel('average_rewards')
plt.plot(average_scores)
plt.ioff()
plt.show()
