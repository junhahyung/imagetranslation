import numpy as np
from PIL import Image

imdir = '/home/junhahyung/dataset/data/celeba/Img/img_align_celeba/000001.jpg'
imdirhq = '/home/junhahyung/dataset/data/celeba/Img/img_align_celeba_hq/000001.jpg'

img = Image.open(imdir)
img = np.asarray(img)
imghq = Image.open(imdirhq)
imghq = np.asarray(imghq)

print(img.shape)
print(imghq.shape)
