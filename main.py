import random

import numpy as np

import gym
import tensorflow as tf
from collections import deque

GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of miniBatch


class DQN():
    # DQN Agent
    def __init__(self, env):
        self.replay_buf = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = not env.action_space.n  # dimension（维度）

        self.create_Q_netword()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_netword(self):
        # network weights
        w1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])

        w2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layer
        h_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)  # mat multiply 矩阵相乘
        # Q layer
        self.Q_value = tf.matmul(h_layer, w2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_mean(tf.matmul(self.Q_value, self.action_input), reduction_indices=1)  # Q值更新公式，新值-旧值
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buf.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buf) > REPLAY_SIZE:
            self.replay_buf.popleft()

        if len(self.replay_buf) > BATCH_SIZE:
            self.train_Q_netword()

    def train_Q_netword(self):
        self.time_step += 1

        # step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buf, BATCH_SIZE)

        # 横向开始取值 state, action, reward, next_state, done
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # step 2:calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})

        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def egreedy_action(self, state):
        Q_value = \
        self.Q_value.eval(feed_dict={  # feed_dict的作用，结合placeholder来看，他的作用是填充前面占位符的定义的数据。比如Q_value的计算调用了state_input
            self.state_input: [state]
        })[0]

        if random.random() <= self.epsilon:
            return random.random().randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)  # 产生正态分布的随机数，用来初始化神经网络的参数， shape是维度，可以是一个列表。例如[2, 3]代表2行3列
        return tf.Variable(initial)  # tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


# ------------------------------------------------------------------------------------------

ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test very 100 episode

def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            reward_agent = -1 if done else 0.1

            agent.perceive(state, action, reward, next_state, done)  # 相当于别的程序里的RL.learn
            state = next_state
            if done:
                break

            if episode % 100 == 0:
                total_raward = 0
                for i in range(TEST):
                    state = env.reset()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_raward += reward
                    if done:
                        break
                ave_reward = total_raward/TEST
                print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
                if ave_reward >= 200:
                    break
if __name__ == '__main__':
    main()