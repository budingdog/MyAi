import gym
from dqn import DQN
from dqn_config import DqnConfig
import time
from collections import deque
from sample_buffer import SampleBuffer
import pickle
from img_processor import ImgProcessor

# ENV_NAME = 'CartPole-v0'
# ENV_NAME = 'MsPacman-v0'
ENV_NAME = 'SpaceInvaders-v0'
# ENV_NAME = 'Breakout-v0'
EPISODE = 10000  # Episode limitation
STEP = 10000  # Step limitation in an episode
CHECK_POINT_STEP = 10
TEST = 3  # The number of experiment test every 100 episode
DISP_DELAY = 0
VERSION = 'si-v0'
FRAME = 4
TRAIN_LOOP = 50


class Rewarder(object):

    def __init__(self):
        self.last_info = None

    def get_reward(self, reward, done, info):
        res_result = 0
        if done == False:
            if reward > 0:
                res_result = 1.0
            elif self.last_info != None and self.last_info['ale.lives'] - info['ale.lives'] > 0:
                res_result = -1.0
        else:
            if info['ale.lives'] > 0:
                res_result = 1.0
            else:
                res_result = -1.0
        self.last_info = info
        return res_result


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    print('game:{} action_space:{}'.format(ENV_NAME, env.action_space.n))
    config = DqnConfig(EPISODE, VERSION)
    agent = DQN(env, config)
    ip = ImgProcessor(config)

    for episode in xrange(1, EPISODE):
        # initialize task
        state = env.reset()
        state = ip.convert(state)
        sb = SampleBuffer(FRAME, [state.shape[0], state.shape[1]])
        input_state = sb.store(state)

        # Train
        reward_sum = 0
        rewarder = Rewarder()
        for step in xrange(STEP):
            action = agent.egreedy_action(input_state, episode)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            next_state = ip.convert(next_state)
            next_input_state = sb.store(next_state)
            reward = rewarder.get_reward(reward, done, _)
            # Define reward for agent
            reward_sum += reward
            agent.store_sample(input_state, action, reward, next_input_state, done)
            # print('step:{}, reward:{}, action:{}'.format(step, reward, action))
            input_state = next_input_state
            if done:
                step_reward = float(reward_sum) / step
                print('episode:{}, step:{}, reward:{}, reward_avg:{}'.format(episode, step, reward_sum,
                                                                             step_reward))
                agent.do_train(TRAIN_LOOP, step, reward_sum)
                break
        # save model
        if episode % CHECK_POINT_STEP == 00:
            agent.save_model()



if __name__ == '__main__':
    main()
