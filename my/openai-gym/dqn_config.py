class DqnConfig(object):

    REPLAY_SIZE = 1024  # experience replay buffer size
    BATCH_SIZE = 128  # size of minibatch
    INITIAL_EPSILON = 0.5  # starting value of epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    VERSION = 'v0'
    CHECK_POINT_PATH = 'model/{}/model.ckpt'.format(VERSION)
    LOG_PATH = 'log/{}'.format(VERSION)
    HIDDEN = [128, 64, 32]
    BATCH_SIZE = 128  # size of minibatch
    GAMMA = 0.9  # discount factor for target Q
    EPISODE = 10000  # Episode limitation

    def __init__(self, episode, version):
        super(DqnConfig, self).__init__()
        self.EPISODE = episode
        self.VERSION = version