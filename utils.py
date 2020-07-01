from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
import torch
import numpy as np
from PIL import Image
import cv2
from skimage.io import imsave
from skimage.measure import regionprops, label
from skimage.segmentation import watershed,find_boundaries
from skimage.morphology import closing, dilation, disk, opening, erosion, remove_small_holes, remove_small_objects
import matplotlib.pyplot as plt

class MyDataSet(Dataset):
    def __init__(self, input_data, mask_data=None, p=0.5, type = 'train'):  # 第一步初始化各个变量
        self.input_data = input_data #导入所有数据
        self.mask_data = mask_data
        self.type = type
        self.p = p
    def __getitem__(self, idx):  # 获取数据
        img = self.input_data[idx]
        if self.type == 'test':
            x = self.test_data_augmentation(img)
            return x
        else:
            mask = self.mask_data[idx]
            x,y = self.train_data_augmentation(self.p, img, mask)
            return x,y
    def __len__(self):
        return len(self.input_data)  # 返回数据集长度

    # 训练集数据增强
    def train_data_augmentation(self, p, x, y):
        dim_x = x.shape[-1]
        dim_y = y.shape[-1]
        mask_size = x.shape[0] + 12
        new_x = torch.empty((dim_x, mask_size, mask_size))
        new_y = torch.empty((dim_y, mask_size, mask_size))
        hflip, vflip = False, False
        angle = 0
        top,left = 0, 0
        if np.random.rand() > p:
            hflip = True
        if np.random.rand() > p:
            vflip = True
        if np.random.rand() > 0:
            angle = np.random.randint(-30,30)
        if np.random.rand() > p:
            top,left = np.random.randint(0,8), np.random.randint(0,8)
        for i in range(dim_x):
            tmp_x = x[:,:,i:i+1]
            tmp_img = tf.to_pil_image((tmp_x))
            tmp_img = tf.pad(tmp_img, 6)
            if hflip:
                tmp_img = tf.hflip(tmp_img)
            if vflip:
                tmp_img = tf.vflip(tmp_img)
            tmp_img = tf.pad(tmp_img, 4)
            tmp_img = tf.crop(tmp_img, top, left, mask_size, mask_size)
            tmp_img = tmp_img.convert('P')
            tmp_img= tf.rotate(tmp_img, angle)
            new_x[i:i+1,:,:] = tf.to_tensor(tmp_img)
        for i in range(dim_y):
            tmp_y = y[:,:,i:i+1]
            tmp_img = tf.to_pil_image((tmp_y))
            tmp_img = tf.pad(tmp_img, 6)
            if hflip:
                tmp_img = tf.hflip(tmp_img)
            if vflip:
                tmp_img = tf.vflip(tmp_img)
            tmp_img = tf.pad(tmp_img, 4)
            tmp_img = tf.crop(tmp_img, top, left, mask_size, mask_size)
            tmp_img = tmp_img.convert('P')
            tmp_img = tf.rotate(tmp_img, angle)
            new_y[i:i+1,:,:] = tf.to_tensor(tmp_img)
        return new_x, new_y

    # 测试集数据处理
    def test_data_augmentation(self, x):
        dim = x.shape[-1]
        mask_size = x.shape[0]+12
        new_x = torch.empty((dim, mask_size, mask_size))
        for i in range(dim):
            tmp_x = x[:,:,i]
            tmp_img = tf.to_pil_image((tmp_x))
            tmp_img = tf.pad(tmp_img, 6)
            new_x[i,:,:] = tf.to_tensor(tmp_img)
        return new_x

# dataset1 生成gt
# 返回2通道的mask，1通道为分水岭的核，2通道为完整细胞
def generate_mask(labels):
    border = find_boundaries(labels)
    marker = (255*(labels > 0)).astype(np.uint8)
    marker2 = np.where(border, 0, marker)
    marker2 = erosion(marker2, disk(3))
    for i in range(1,np.max(labels)+1):
        temp = (labels == i).astype(np.uint8)
        if not (temp*marker2).any():
            marker2 = marker2 + temp
    mask = np.stack((marker2, marker, labels.astype(np.uint8)),axis=-1)
    return mask

# dataset2 生成gt
# 返回2通道的mask，1通道为分水岭的核，2通道为边界
def generate_mask2(labels):
    border = find_boundaries(labels, mode='inner')
    marker = (255*(labels > 0)).astype(np.uint8)
    marker = np.where(border, 0, marker)
    marker = erosion(marker, disk(5))
    mask = np.stack((marker, border, labels.astype(np.uint8)),axis=-1)
    return mask

# 计算二分类的Jaccard相似度
def cal_jaccard_binary(y_pred, y_true, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred>threshold).float()
    y_true = (y_true > 0)
    inter = torch.sum((y_true + y_pred) == 2)
    union = torch.sum((y_true + y_pred) >= 1)
    jaccard_similarity =  float(inter)/(float(union) + 1e-7)
    return jaccard_similarity

# 计算Jaccard score
def cal_jaccard(real, pred):
    label_num = np.max(real)
    score = 0
    num = 0
    for i in range(label_num + 1):
        pos_i = (real == i)
        a = np.bincount(pred[pos_i])
        pred_label = 1000000
        if a.shape[0] > 0: pred_label = np.argmax(a)
        pred_tmp = np.where(pred == pred_label, 1, 0)
        real_tmp = np.where(real == i, 1, 0)
        if real_tmp.any():
            num += 1
            js = np.count_nonzero((pred_tmp * real_tmp) == 1)/np.count_nonzero((pred_tmp + real_tmp) >= 1)
            score += js
    return score/num

# 图片读入处理
def unit16b2uint8(img):
    if img.dtype == 'uint8':
        return img
    elif img.dtype == 'uint16':
        return img.astype(np.uint8)
    else:
        raise TypeError('No such of img transfer type: {} for img'.format(img.dtype))

# 输入标准化
def img_standardization(img):
    img = unit16b2uint8(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
        return img
    elif len(img.shape) == 3:
        return img
    else:
        raise TypeError('The Depth of image large than 3 \n')

# dataset1 marker后处理
def marker_postprocess(marker, threshold=0.7):
    # 开闭运算分别去除暗细节和亮的细节
    marker0 = opening(marker,disk(5))
    marker1 = closing(marker0,disk(5))
    # 二值化
    marker1 = (marker1>threshold)
    marker_temp = label(marker1)
    # 腐蚀
    marker2 = erosion(marker1, disk(9))
    marker3 = remove_small_holes(marker2)
    #补回之前被腐蚀掉的种子
    for i in range(1,np.max(marker_temp)+1):
        temp = (marker_temp == i).astype(np.uint8)
        if not (temp*marker3).any():
            marker3 = marker3 + temp
    markers = label(marker3)
    return markers

# dataset1 mask后处理
def mask_postprocess(mask, threshold=0.4):
    mask1 = (mask > threshold).astype(np.uint8)
    mask2 = remove_small_holes(mask1)
    return mask2


def mini_postprocess(final_mask, origin_mask):
    # 添加那些被移除的小细胞
    water_mask = (final_mask > 0).astype(np.uint8)
    dropped = origin_mask - water_mask
    dropped = label(dropped)
    dropped = np.where(dropped, dropped + final_mask.max(), 0)
#     final_mask1 = final_mask + dropped
    # 去除那些很小的杂点
    # remove_mask = remove_small_objects(final_mask, 100)
    out = final_mask + dropped
    return out
# dataset1 后处理
def output_postprocess(marker, border):
    # 得到种子点
    marker_new = marker_postprocess(marker)
    # 得到mask
    mask_all = mask_postprocess(border)
    # 让有label的地方是最小值
    water_all = 255 - marker_new
    # 分水岭
    out = watershed(water_all, marker_new, mask=mask_all)
    # out = mini_postprocess(out, mask_all)
    return out

# dataset2 marker后处理
def marker_postprocess2(marker, threshold=0.8):
    #二值化
    marker1 = (marker>threshold).astype(np.uint8)
    marker2 = remove_small_holes(marker1)
    # 开运算
    marker3 = opening(marker2, disk(12))
    labels = label(marker3)
    return labels

# dataset2 border后处理
def border_postprocess2(border, threshold=0.08):
    border1 = (border > threshold)
    border1 = border1.astype(np.uint8)
    # border2 = closing(border1)
    return border1

# dataset2 后处理
def output_postprocess2(marker,border):
    marker_new = marker_postprocess2(marker)
    border_new = border_postprocess2(border)
    mask_all = (marker > 0.5).astype(np.uint8)
    mask_all = np.maximum(mask_all, border_new)
    mask_all = remove_small_holes(mask_all, 100)
    # 让有label的地方是最小值
    water_all = 255 - marker_new
    out = watershed(water_all, marker_new, mask=mask_all)
    out = dilation(out, disk(5))
    return out

# 保存图片
def save_image(image,idx, type):
    num = '000'
    if idx < 10:
        num = '00' + str(idx)
    elif idx <  100:
        num = '0' + str(idx)
    path = './test_RES/' + type + num + '.png'
    # image = image.astype(np.uint16)
    imsave(path, image)
    return

# 可视化
def visual(img, gt):
    img = cv2.imread(img, -1)
    gt = cv2.imread(gt, -1)
    label = np.unique(gt)
    height, width = img.shape[:2]
    visual_img = np.zeros((height, width, 3))
    for lab in label:
        if lab == 0:
            continue
        color = np.random.randint(low=0, high=255, size=3)
        visual_img[gt==lab, :] = color
    return img.astype(np.uint8), visual_img.astype(np.uint8)
