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
DISP_DELAY = 0


class Rewarder(object):

    def __init__(self):
        self.last_info = None

    def get_reward(self, reward, done, info):
        res_result = 0
        if done == False:
            if reward > 0:
                res_result = reward
            elif self.last_info != None and self.last_info['ale.lives'] - info['ale.lives'] > 0:
                res_result = -50
        else:
            if info['ale.lives'] > 0:
                res_result = 1000
            else:
                res_result = -100
        self.last_info = info
        return res_result


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    config = DqnConfig(EPISODE, 'v0')
    config.HIDDEN = [128, 64, 32]
    agent = DQN(env, config)

    for episode in xrange(1, EPISODE):
        # initialize task
        state = env.reset()
        # Train
        reward_sum = 0;
        rewarder = Rewarder()
        for step in xrange(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            reward = rewarder.get_reward(reward, done, _)
            # Define reward for agent
            reward_sum += reward
            agent.perceive(state, action, reward, next_state, done)
            # print('step:{}, reward:{}, action:{}'.format(step, reward, action))
            state = next_state
            if done:
                step_reward = float(reward_sum) / step
                print('episode:{}, step:{}, reward:{}, reward_avg:{}'.format(episode, step, reward_sum,
                                                                             step_reward))
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
            avg_reward = total_reward / TEST
            print '\nepisode: ', episode, 'Evaluation Average Reward:', avg_reward


if __name__ == '__main__':
    main()
