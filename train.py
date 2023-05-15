import torch
import torch.nn as nn
import models.resnet as ResNet
import models.senet as SENet
import utils
import torch.backends.cudnn as cudnn
import numpy as np
import time
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

from dataset.vggface2triplet import VGGFaces2


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# net = ResNet.resnet50(num_classes=8631, include_top=False)
# utils.load_state_dict(net, 'models/resnet50_ft_weight.pkl')
net = SENet.senet50(num_classes=8631, include_top=False)
utils.load_state_dict(net, 'models/senet50_ft_weight.pkl')
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
net.eval()

print('==> Preparing data..')


import pandas as pd
import csv
def get_id_label_map(meta_file):
    N_IDENTITY = 9131  # total number of identities in VGG Face2
    N_IDENTITY_PRETRAIN = 8631  # the number of identities used in training by Caffe
    identity_list = meta_file
    df = pd.read_csv(identity_list, sep=',\s+', quoting=csv.QUOTE_ALL, encoding="utf-8")
    df["class"] = -1
    df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY_PRETRAIN)
    df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY_PRETRAIN, N_IDENTITY)
    # print(df)
    key = df["Class_ID"].values
    val = df["class"].values
    id_label_dict = dict(zip(key, val))
    return id_label_dict


val_dir = '../../estudo/data/VGG-Face2/data/test'
train_dir = '../../estudo/data/VGG-Face2/data/train'
val_list_file = '../../estudo/data/VGG-Face2/data/test_list.txt'
train_list_file = '../../estudo/data/VGG-Face2/data/train_list.txt'
meta = '../../estudo/data/VGG-Face2/meta/identity_meta.csv'
id_meta = get_id_label_map(meta)
data_val = VGGFaces2(val_dir, val_list_file, id_meta, split='valid')
# data_train = VGGFaces2(train_dir, train_list_file, id_meta, split='train')


anchor, positive, negative = data_val[1]

a_img = anchor[0]
p_img = positive[0]
n_img = negative[0]

img = a_img.numpy()
img = img.transpose(1, 2, 0)
# img = img.astype(np.uint8)
img = img[:, :, ::-1]
plt.imshow(img)
plt.axis('off')
plt.show()

img = p_img.numpy()
img = img.transpose(1, 2, 0)
# img = img.astype(np.uint8)
img = img[:, :, ::-1]
plt.imshow(img)
plt.axis('off')
plt.show()

img = n_img.numpy()
img = img.transpose(1, 2, 0)
# img = img.astype(np.uint8)
img = img[:, :, ::-1]
plt.imshow(img)
plt.axis('off')
plt.show()
exit()

torch.set_grad_enabled(False)
print('\n\n')
print('########################################')

