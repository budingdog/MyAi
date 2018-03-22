from PIL import Image
import numpy as np

class ImgProcessor(object):

    def __init__(self):
        super(ImgProcessor, self).__init__()

    def convert(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((84, 84), Image.ANTIALIAS)
        return np.array(img)