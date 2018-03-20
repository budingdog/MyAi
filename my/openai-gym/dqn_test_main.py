import gym
from dqn import DQN
from dqn_config import DqnConfig
import time
from sample_buffer import SampleBuffer

# ENV_NAME = 'CartPole-v0'
# ENV_NAME = 'MsPacman-ram-v0'
ENV_NAME = 'MsPacman-v0'
# ENV_NAME = 'SpaceInvaders-ram-v0'
EPISODE = 10000  # Episode limitation
STEP = 300000  # Step limitation in an episode
CHECK_POINT_STEP = 10
TEST = 10  # The number of experiment test every 100 episode
DISP_DELAY = 30
VERSION = 'v5'
FRAME = 4

def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    config = DqnConfig(EPISODE, VERSION)
    agent = DQN(env, config)

    for episode in xrange(EPISODE):

        total_reward = 0
        for i in xrange(TEST):
            state = env.reset()
            sb = SampleBuffer(FRAME, len(state))
            input_state = sb.store(state)
            for j in xrange(STEP):
                env.render()
                time.sleep(DISP_DELAY / 1000.0)
                action = agent.action(input_state)  # direct action for test
                state, reward, done, _ = env.step(action)
                input_state = sb.store(state)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print '\nepisode: ', episode, 'Evaluation Average Reward:', ave_reward


if __name__ == '__main__':
    main()