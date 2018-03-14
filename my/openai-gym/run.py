

total_reward = 0
for i in xrange(TEST):
    state = env.reset()
    for j in xrange(STEP):
        env.render()
        action = agent.action(state)  # direct action for test
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
ave_reward = total_reward / TEST
print '\nepisode: ', episode, 'Evaluation Average Reward:', ave_reward