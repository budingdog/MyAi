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
        self.state_height = env.observation_space.shape[0]
        self.state_width = env.observation_space.shape[1]
        self.state_channel = env.observation_space.shape[2]
        self.action_dim = env.action_space.n

        # Init session
        self.session = tf.InteractiveSession()
        self.create_Q_network()
        self.create_training_method()
        self.writer = tf.summary.FileWriter(config.LOG_PATH, self.session.graph)
        self.create_summary()

        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()

        try:
            self.saver.restore(self.session, config.CHECK_POINT_PATH)
        except:
            self.session.run(tf.global_variables_initializer())

    def __del__(self):
        self.session.close()
        self.writer.close()

    def create_Q_network(self):
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_height, self.state_width, self.state_channel,
                                                    self.config.FRAME])
        with tf.name_scope('image-compress'):
            compress = tf.constant(np.ones([1, 5, 5, 3, 1]), dtype=tf.float32)
            compressed_input = tf.nn.conv3d(self.state_input, filter=compress, strides=[1, 3, 2, 3, 1], padding='SAME')
        with tf.name_scope('Q-network'):
            conv1 = tf.get_variable('conv1',
                                    [self.state_height, self.state_width, self.state_channel, self.config.HIDDEN[0]],
                                    dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer)
            conv_layer1 = tf.nn.conv2d(compressed_input, filter=conv1, strides=[1, 8, 8, 1], padding='SAME',
                                       name='conv_layer1')
            conv2 = tf.get_variable('conv2',
                                    [self.state_height, self.state_width, self.config.HIDDEN[0], self.config.HIDDEN[1]],
                                    dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer)
            conv_layer2 = tf.nn.conv2d(conv_layer1, filter=conv2, strides=[1, 8, 8, 1], padding='SAME',
                                       name='conv_layer2')
            input_tensors = tf.reshape(conv_layer2,
                                       [-1, conv_layer2.shape[1] * conv_layer2.shape[2] * conv_layer2.shape[3]])
            for i in range(2, len(self.config.HIDDEN)):
                input_tensors = tf.layers.dense(input_tensors, units=self.config.HIDDEN[i], activation=tf.nn.relu,
                                                name='hidden{}'.format(i))
            self.Q_value = tf.layers.dense(input_tensors, units=self.action_dim, activation=tf.nn.sigmoid,
                                           name='output')

    def create_training_method(self):
        with tf.name_scope('loss'):
            self.action_input = tf.placeholder("float", [None, self.action_dim], name='action')  # one hot presentation
            self.y_input = tf.placeholder("float", [None], name='y_label')
            Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1, name='y')
            self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
            # self.cost = tf.losses.softmax_cross_entropy(self.y_input, Q_action)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def create_summary(self):
        tf.summary.scalar('Q-value', tf.reduce_max(self.Q_value))
        tf.summary.scalar('loss', self.cost)

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
        print('step:{}'.format(self.time_step))
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
        tf.nn.conv2d
