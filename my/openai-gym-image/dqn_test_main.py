import gym
from dqn import DQN
from dqn_config import DqnConfig
import time
from collections import deque
from sample_buffer import SampleBuffer
import pickle
from img_processor import ImgProcessor

# ENV_NAME = 'CartPole-v0'
ENV_NAME = 'MsPacman-v0'
# ENV_NAME = 'SpaceInvaders-v0'
# ENV_NAME = 'Breakout-v0'
EPISODE = 1000  # Episode limitation
STEP = 300000  # Step limitation in an episode
CHECK_POINT_STEP = 1
TEST = 3  # The number of experiment test every 100 episode
DISP_DELAY = 20
VERSION = 'pac-v4'
FRAME = 4


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    config = DqnConfig(EPISODE, VERSION)
    agent = DQN(env, config)
    ip = ImgProcessor()

    for episode in xrange(1, EPISODE):
        # initialize task
        state = env.reset()
        state = ip.convert(state)
        sb = SampleBuffer(FRAME, [state.shape[0], state.shape[1]])
        input_state = sb.store(state)

        total_reward = 0
        for i in xrange(TEST):
            for j in xrange(STEP):
                env.render()
                time.sleep(DISP_DELAY / 1000.0)
                action = agent.action(input_state)  # direct action for test
                next_state, reward, done, _ = env.step(action)
                next_state = ip.convert(next_state)
                next_input_state = sb.store(next_state)
                total_reward += reward
                input_state = next_input_state
                if done:
                    break
        avg_reward = total_reward / TEST
        print '\nepisode: ', episode, 'Evaluation Average Reward:', avg_reward


if __name__ == '__main__':
    main()
