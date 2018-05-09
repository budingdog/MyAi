from PIL import Image
import numpy as np


class ImgProcessor(object):

    def __init__(self):
        super(ImgProcessor, self).__init__()

    def convert(self, state):
        state = state[1:171]
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((84, 84), Image.ANTIALIAS)
        state = np.array(img)
        return state