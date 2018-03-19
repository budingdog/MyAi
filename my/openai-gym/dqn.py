import random
from collections import deque
import tensorflow as tf
import numpy as np


class DQN(object):
    # DQN Agent
    def __init__(self, env, config):

        self.config = config
        # init experience replay
        self.replay_buffer = deque(maxlen=config.REPLAY_SIZE / 2)
        self.replay_buffer_neg = deque(maxlen=config.REPLAY_SIZE / 2)
        # init some parameters
        self.time_step = 0
        self.epsilon = config.INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.writer = tf.summary.FileWriter(config.LOG_PATH, self.session.graph)
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()

        try:
            self.saver.restore(self.session, config.CHECK_POINT_PATH)
        except:
            self.session.run(tf.initialize_all_variables())

    def __del__(self):
        self.session.close()
        self.writer.close()

    def create_Q_network(self):
        with tf.name_scope('Q-network'):
            # input layer
            self.state_input = tf.placeholder("float", [None, self.state_dim])
            input_dim = self.state_dim
            input_tensors = self.state_input

            # hidden layers
            len_hidden = len(self.config.HIDDEN)
            for i in range(0, len_hidden):
                W = self.weight_variable([input_dim, self.config.HIDDEN[i]])
                B = self.bias_variable([self.config.HIDDEN[i]])
                h_layer = tf.nn.relu(tf.matmul(input_tensors, W) + B)
                input_tensors = h_layer
                input_dim = self.config.HIDDEN[i]
            # Q Value layer
            W_q = self.weight_variable([input_dim, self.action_dim])
            B_q = self.bias_variable([self.action_dim])
            self.Q_value = tf.matmul(h_layer, W_q) + B_q
            tf.summary.scalar('Q-value', tf.reduce_max(self.Q_value))

    def create_training_method(self):
        with tf.name_scope('loss'):
            self.action_input = tf.placeholder("float", [None, self.action_dim], name='action')  # one hot presentation
            self.y_input = tf.placeholder("float", [None], name='y_label')
            Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1, name='y')
            self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
            # self.cost = tf.losses.softmax_cross_entropy(self.y_input, Q_action)
            tf.summary.scalar('loss', self.cost)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        if reward != 0:
            self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        else:
            self.replay_buffer_neg.append((state, one_hot_action, reward, next_state, done))

        if (len(self.replay_buffer) + len(self.replay_buffer_neg)) > self.config.BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        action_batch, minibatch, next_state_batch, reward_batch, state_batch = self.obtain_minibatch()

        # Step 2: calculate y
        y_batch = self.generate_y_label(minibatch, next_state_batch, reward_batch)

        summary, opt = self.session.run([self.summary, self.optimizer], feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })
        self.writer.add_summary(summary, self.time_step)

    def generate_y_label(self, minibatch, next_state_batch, reward_batch):
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, self.config.BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.config.GAMMA * np.max(Q_value_batch[i]))
        return y_batch

    def obtain_minibatch(self):
        minibatch = random.sample(list(self.replay_buffer) + list(self.replay_buffer_neg), self.config.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        return action_batch, minibatch, next_state_batch, reward_batch, state_batch

    def egreedy_action(self, state):
        self.epsilon -= (self.config.INITIAL_EPSILON - self.config.FINAL_EPSILON) / self.config.EPISODE
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return self.action(state)

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def save_model(self):
        self.saver.save(self.session, self.config.CHECK_POINT_PATH)
