import gym
from dqn import DQN
from dqn_config import DqnConfig
import time

# ENV_NAME = 'CartPole-v0'
ENV_NAME = 'MsPacman-ram-v0'
# ENV_NAME = 'SpaceInvaders-ram-v0'
EPISODE = 10000  # Episode limitation
STEP = 300000  # Step limitation in an episode
CHECK_POINT_STEP = 10
TEST = 3  # The number of experiment test every 100 episode
DISP_DELAY = 30

def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env, DqnConfig(EPISODE, 'v5'))

    for episode in xrange(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        reward_sum = 0;
        for step in xrange(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -100
            # Define reward for agent
            reward_sum += reward
            agent.perceive(state, action, reward, next_state, done)
            # print('step:{}, reward:{}, action:{}'.format(step, reward, action))
            state = next_state
            if done:
                print('episode:{}, step:{}, reward:{}, reward_avg:{}'.format(episode, step, reward_sum,
                                                                             float(reward_sum) / step))
                break
        # save model
        if episode % CHECK_POINT_STEP == 0:
            agent.save_model()
        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                for j in xrange(STEP):
                    env.render()
                    time.sleep(DISP_DELAY / 1000.0)
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print '\nepisode: ', episode, 'Evaluation Average Reward:', ave_reward


if __name__ == '__main__':
    main()