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

from models.vggface2 import VGGFaces2


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


val_dir = '../../estudo/data/test'
val_list_file = '../../estudo/data/test_list.txt'
val_meta = '../../estudo/data/meta/identity_meta.csv'
val_id_meta = get_id_label_map(val_meta)
validation = VGGFaces2(val_dir, val_list_file, val_id_meta, split='valid')


img, label, img_file, class_id = validation[1]

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_2 = transforms.Compose([
    transforms.ToTensor(),
])

mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
# mean_bgr = np.array([0, 0, 0])  # from resnet50_ft.prototxt


def transform(img):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= mean_bgr
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img


def untransform(img):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img += mean_bgr
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    return img



def get_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img = torchvision.transforms.Resize(256)(img)
    img = torchvision.transforms.Resize(224)(img)
    img = torchvision.transforms.CenterCrop(224)(img)
    img = np.array(img, dtype=np.uint8)
    assert len(img.shape) == 3  # assumes color images and no alpha channel
    return img

torch.set_grad_enabled(False)
print('\n\n')
print('########################################')

im1 = get_img("im1.jpg")
im2 = get_img("im2.jpg")
im3 = get_img("im3.jpg")


im1 = transform(im1)
im2 = transform(im2)
im3 = transform(im3)


t = time.time()
pred1 = net(im1.unsqueeze(0)).squeeze()
pred2 = net(im2.unsqueeze(0)).squeeze()
pred3 = net(im3.unsqueeze(0)).squeeze()
print('total time: {} ms'.format(round(1000 * (time.time() - t), 2)))


print('\nEmbedding size:', pred1.size()[0])
# p1 = F.softmax(net(im1.unsqueeze(0)), dim=1)
# bp1 = torch.argmax(p1, dim=1)
# p2 = F.softmax(net(im2.unsqueeze(0)), dim=1)
# bp2 = torch.argmax(p2, dim=1)
# p3 = F.softmax(net(im3.unsqueeze(0)), dim=1)
# bp3 = torch.argmax(p3, dim=1)
#
# pred1 = F.normalize(pred1, dim=0)
# pred2 = F.normalize(pred2, dim=0)
# pred3 = F.normalize(pred3, dim=0)


cosine_distance12 = (F.cosine_similarity(pred1, pred2, dim=0) + 1)/2
cosine_distance13 = (F.cosine_similarity(pred1, pred3, dim=0) + 1)/2
cosine_distance23 = (F.cosine_similarity(pred2, pred3, dim=0) + 1)/2

print('Distance 1-2:', cosine_distance12)
print('Distance 1-3:', cosine_distance13)
print('Distance 2-3:', cosine_distance23)


cv2.imwrite('teste1.jpg', untransform(im1))
cv2.imwrite('teste2.jpg', untransform(im2))
cv2.imwrite('teste3.jpg', untransform(im3))