import pickle
from img_processor import ImgProcessor
from PIL import Image
import numpy as np

img_array = pickle.load(open('data/img_array'))
ip = ImgProcessor()
new = ip.convert(img_array)
new = new * 256 + 128
new = new.astype(np.int32)
img_new = Image.fromarray(new)
img_new.convert('RGB').save('data/myimg_4.jpeg')