import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
from losses import *
from utils import *
from UNet import *
from  UNetplusplus import *
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 读取数据
train_x_path = './supplementary_modify/dataset1/train/'
train_y_path = './supplementary_modify/dataset1/train_GT/SEG/'
test_x_path = './supplementary_modify/dataset1/test/'
train_x = []
train_y = []
test_x = []
filename_train_x = os.listdir(train_x_path)
train_num = len(filename_train_x)
for i in tqdm(range(train_num)):
    filename = filename_train_x[i]
    img = cv2.imread(train_x_path + filename, -1)
    img = img_standardization(img)
    train_x.append(img)
    filename2 = 'man_seg' + filename[1:]
    img_label = cv2.imread(train_y_path + filename2, -1)
    img_mask = generate_mask(img_label)
    train_y.append(img_mask)
filename_test_x = os.listdir(test_x_path)
test_num = len(filename_test_x)
for i in tqdm(range(test_num)):
    filename = filename_test_x[i]
    img_test = cv2.imread(test_x_path + filename, -1)
    img_test = img_standardization(img_test)
    test_x.append(img_test)
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test_x = np.asarray(test_x)


# 划分训练集和验证集
BATCH_SIZE = 1
sample_num = train_x.shape[0] #样本数量
pic_size = train_x.shape[1]
train_num = int(sample_num*0.8)
val_num = int(sample_num - train_num)

train_data = MyDataSet(train_x, train_y)
test_data = MyDataSet(test_x)
train, val = random_split(train_data, [train_num, val_num])
train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
# tmp = train_data[0][1]
# print(cal_jaccard_similarity(tmp, tmp))

# 超参数设置
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_channel = 2
in_channel = 1
model = U_Net(in_channel,out_channel, 'UNet')
deep_supervision=True
lr = 0.0001
# model = UNetPlusPlus(in_channel, out_channel, deep_supervision)
model.apply(init_weights)
model = model.to(device)
# summary(model, input_size=(1, 628, 628))
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss = nn.BCEWithLogitsLoss()
loss = FocalDiceLoss(2, 0.25)
# loss = BCEDiceLoss()
# loss 停止下降时改变学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
num_epochs = 20

for epoch in range(num_epochs):
    train_loader = tqdm(train_loader)
    train_loss = 0
    model.train()
    for i, (X, Y) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
        X = Variable(X).to(device) #转化数据类型
        #X = Variable(X)
        X = X.float()
        Y = Variable(Y).to(device)
        Y = Y.float()
        outs = model(X)  # 正向传播
        # out1 = model1(X)
        # lossvalue = loss(outs, Y[:,0:out_channel,:,:])  # 求损失值
        # out2 = model2(X)
        lossvalue = 0
        j = 0
        # for out in outs:
        # # for i in range(len(outs)):
        #     lossvalue += w[j]* loss(out, Y[:,0:out_channel,:,:])  # 求损失值
        #     j += 1
        lossvalue += loss(outs, Y[:,0:out_channel,:,:])
        # lossvalue/len(outs)
        optimizer.zero_grad()  # 优化器梯度归零
        lossvalue.backward()  # 反向转播，刷新梯度值
        nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()  # 优化器运行一步
        # 计算损失
        train_loss += float(lossvalue)
    val_loss = 0  # 定义验证损失
    val_js = 0
    max_js = 0
    model.eval() #模型转化为评估模式
    val_loader = tqdm(val_loader)
    for i,(X, Y) in enumerate(val_loader):
        X = Variable(X).to(device)
        X = X.float()
        Y = Variable(Y).to(device)
        with torch.no_grad():
            outs = model(X)
        lossvalue = 0
        j = 0
        # for out in outs:
        #     lossvalue += w[j]*loss(out, Y[:,0:out_channel,:,:])  # 求损失值
        #     new_w[j] += cal_jaccard_similarity(out, Y[:,0:out_channel,:,:])
        #     j += 1
        lossvalue += loss(outs, Y[:,0:out_channel,:,:])
        # lossvalue/len(outs)
        # lossvalue = loss(outs, Y[:,0:out_channel,:,:])  # 求损失值
        val_loss += float(lossvalue)
        score = cal_jaccard_binary(outs, Y[:,0:out_channel,:,:])
        val_js += score
    scheduler.step(val_js)
    max_js = max(max_js,val_js)
    if max_js == val_js:
        torch.save(model, 'modelunet2.pkl')
    print("epoch:" + ' ' + str(epoch))
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print("lr:", lr)
    print("train lose:" + ' ' + str(train_loss / len(train_loader)))
    print("val lose:" + ' ' + str(val_loss / len(val_loader)))
    print("val js:" + ' ' + str(val_js / len(val_loader)))

val_marker_x = []
# val_border_x = []
val_marker_y = []
# val_border_y = []

for i, (X,Y) in enumerate(val_loader):
    X = Variable(X).to(device)
    X = X.float()
    Y = Variable(Y).to(device)
    Y = Y.float()
    with torch.no_grad():
        out = model(X)
        marker = []
        # border = []
        # for out in out:
        out_prob1 = torch.sigmoid(out[:,0,:,:])
            # out_prob2 = torch.sigmoid(out[:,1,:,:])
        marker = out_prob1.cpu().clone().numpy()[0]
            # border.append(out_prob2.cpu().clone().numpy()[0])
    # marker = np.asarray(marker)
    # border = np.asarray(border)
    val_marker_y.append(Y[:,0,:,:].cpu().clone().numpy()[0])
    # val_border_y.append(Y[:,1,:,:].cpu().clone().numpy()[0])
    val_marker_x.append(marker)
    # val_border_x.append(border)

val_marker_x = np.asarray(val_marker_x)
# val_border_x = np.asarray(val_border_x)
val_marker_y = np.asarray(val_marker_y)
# val_border_y = np.asarray(val_border_y)


model = torch.load('modelunet2.pkl')
save_dir = '/content/gdrive/My Drive/Colab Notebooks/Segmentation/dataset2/test_RES1/'
for i, X in enumerate(test_loader):
    X = Variable(X).to(device)
    X = X.float()
    # Y = Variable(Y).to(device)
    # Y = Y.float()
    with torch.no_grad():
        out = model(X)
        out_prob1 = torch.sigmoid(out[:,0,:,:])
        # out_prob2 = torch.sigmoid(out[:,1,:,:])
        # out1 = model1(X)
        # out2 = model2(X)
        # out_prob1 = torch.sigmoid(out1)
        # out_prob2 = torch.sigmoid(out2)
        marker = out_prob1.cpu().clone().numpy()[0]
        # marker, border = out_prob1.cpu().clone().numpy()[0], out_prob2.cpu().clone().numpy()[0]
    # out_img = output_postprocess(marker, border)
    num = '000'
    if i < 10:
        num = '00' + str(i)
    elif i <  100:
        num = '0' + str(i)
    path = './test_RES1/' + 'marker' + num + '.tif'
    path2 = './test_RES1/' + 'border' + num + '.tif'
    # out_img = out_img.astype(np.uint16)
    imsave(path, marker)
    # imsave(path2, border)
