from PIL import Image
import numpy as np


class ImgProcessor(object):

    def __init__(self, config):
        super(ImgProcessor, self).__init__()
        self.config = config

    def convert(self, state):
        state = state[self.config.PLAY_AREA_START: self.config.PLAY_AREA_END]
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((84, 84), Image.ANTIALIAS)
        state = np.array(img)
        state = (state - 128) / 128 - 1
        return state
