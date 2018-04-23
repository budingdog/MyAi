import pickle
from PIL import Image
import numpy as np

img_array = pickle.load(open('data/img_array'))
img_array = img_array[1:170]
img = Image.fromarray(img_array)
img = img.convert('L')
img = img.resize((84, 110),Image.ANTIALIAS)
img.save('data/myimg_2.jpeg')
img_array2 = np.array(img)
pass
