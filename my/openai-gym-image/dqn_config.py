class DqnConfig(object):

    def __init__(self, episode, version):
        super(DqnConfig, self).__init__()
        self.REPLAY_SIZE = 1024  # experience replay buffer size
        self.BATCH_SIZE = 64  # size of minibatch
        self.INITIAL_EPSILON = 0.2  # starting value of epsilon
        self.FINAL_EPSILON = 0.1  # final value of epsilon
        self.VERSION = version
        self.CHECK_POINT_PATH = 'model/{}/model.ckpt'.format(self.VERSION)
        self.CONTEXT_PATH = 'model/{}/context'.format(self.VERSION)
        self.LOG_PATH = 'log/{}'.format(self.VERSION)
        self.HIDDEN = [32, 64, 64, 512]
        self.GAMMA = 0.9  # discount factor for target Q
        self.EPISODE = episode  # Episode limitation
        self.FRAME = 4
        # self.PLAY_AREA_START = 1
        # self.PLAY_AREA_END = 175