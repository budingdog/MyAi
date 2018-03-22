class DqnConfig(object):

    def __init__(self, episode, version):
        super(DqnConfig, self).__init__()
        self.REPLAY_SIZE = 256  # experience replay buffer size
        self.BATCH_SIZE = 32  # size of minibatch
        self.INITIAL_EPSILON = 0.5  # starting value of epsilon
        self.FINAL_EPSILON = 0.01  # final value of epsilon
        self.VERSION = version
        self.CHECK_POINT_PATH = 'model/{}/model.ckpt'.format(self.VERSION)
        self.LOG_PATH = 'log/{}'.format(self.VERSION)
        self.HIDDEN = [16, 32, 256]
        self.BATCH_SIZE = 128  # size of minibatch
        self.GAMMA = 0.9  # discount factor for target Q
        self.EPISODE = episode  # Episode limitation
        self.FRAME = 4