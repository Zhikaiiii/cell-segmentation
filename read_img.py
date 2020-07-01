import os
from skimage.io import imsave, imread
import numpy as np
import matplotlib.pyplot as plt

path1 = './supplementary_modify/Fluo-N2DH-SIM+/01/'
path2 = './supplementary_modify/Fluo-N2DH-SIM+/02/'
path3 = './supplementary_modify/dataset1/train/'
# pic = imread(path)
all_file1 = os.listdir(path1)
all_file1.sort()
all_file2 = os.listdir(path2)
all_file2.sort()
all_pic1 = []
all_pic2 = []
train_pic = []

train_file = os.listdir(path3)
train_file.sort()


for file_name in all_file1:
    file_path = path1 + file_name
    all_pic1.append(imread(file_path))
for file_name in all_file2:
    file_path = path2 + file_name
    all_pic2.append(imread(file_path))
for file_name in train_file:
    file_path = path3 + file_name
    train_pic.append(imread(file_path))

all_idx1 = np.ones(len(all_pic1))
all_idx2 = np.ones(len(all_pic2))
train_idx = np.ones(len(train_pic))

for idx, img in enumerate(all_pic1):
    for idx2, train_img in enumerate(train_pic):
        if not train_idx[idx2]:
            continue
        if (img == train_img).all():
            all_idx1[idx] = 0
            train_idx[idx2] = 0
            break
for idx, img in enumerate(all_pic2):
    for idx2, train_img in enumerate(train_pic):
        if not train_idx[idx2]:
            continue
        if (img == train_img).all():
            all_idx2[idx] = 0
            train_idx[idx2] = 0
            break


#
# origin_pic = os.listdir(path3)
# idx_1 = []
# idx_2 = []
# for filename in os.listdir(path):
#     file = path2 + filename
#     pic = imread(file).astype(np.uint8)
#     file2 = test_x_path + 't' + filename[4:]
#     # pic1 = pic[6:634,6:634]
#     pic2 = imread(file2).astype(np.uint8)
#     # path = './test_RES/mask' + type + '.tif'
#     # image = image.astype(np.uint16)
#     # file2 = path2 + filename
#     # imsave(file2, pic1)
#     plt.imshow(pic)
#     plt.figure()
#     plt.imshow(pic2)
