import torch
import torch.nn as nn
import models.senettriplet as SENet
from src import utils
from src.cosinedistance import CosineDistance
import os
import pandas as pd
import csv
from dataset.vggface2triplet import VGGFaces2
import math
#import time
#time.sleep(1000)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# net = ResNet.resnet50(num_classes=8631, include_top=False)
#if os.path.isfile("best-model-parameters.pt"):
##    print('\nLoad best-model-parameters.py')
#    checkpoint = torch.load("best-model-parameters.pt")
#    net.load_state_dict(checkpoint, strict=True)
#
#print('\nLoad best-model-parameters.py')
#checkpoint = torch.load("best-model-parameters.pt")
#net.load_state_dict(checkpoint, strict=True)
#if device == 'cuda':
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True

print('==> Preparing data..')


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


val_dir = '/dataset/VGG-Face2-crop/data/test/'
train_dir = '/dataset/VGG-Face2-crop/data/train/'
val_list_file = '/dataset/VGG-Face2-crop/meta/test_list.txt'
train_list_file = '/dataset/VGG-Face2-crop/meta/train_list.txt'
meta = '/dataset/VGG-Face2-crop/meta/identity_meta.csv'
id_meta = get_id_label_map(meta)
#data_val = VGGFaces2(val_dir, val_list_file, id_meta, split='valid')
data_train = VGGFaces2(train_dir, train_list_file, id_meta, split='train')

print('\n\n')
print('########################################')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = SENet.senet50(num_classes=8631, include_top=False)
print('\nLoad default wheight (.pkl)')
utils.load_state_dict(net, 'models/senet50_ft_weight.pkl')
if os.path.isfile("best-model-parameters.pt"):
    checkpoint = torch.load("best-model-parameters.pt")
    print('\nLoad best-model-parameters.py')
    net.load_state_dict(checkpoint, strict=True)
net.cuda()


print('##### margin: 0.6 #####')
criterion = nn.TripletMarginWithDistanceLoss(distance_function=CosineDistance(), margin=0.6)
criterion = criterion.cuda()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_loader = torch.utils.data.DataLoader(data_train, num_workers=15, batch_size=34, shuffle=True)
#val_loader = torch.utils.data.DataLoader(data_val, num_workers=4, batch_size=4, shuffle=True)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor_img = anchor[0]
        positive_img = positive[0]
        negative_img = negative[0]

        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)

        optimizer.zero_grad()
        anchor_out, positive_out, negative_out = net(anchor_img, positive_img, negative_img)
        loss = criterion(anchor_out, positive_out, negative_out)
        train_loss = loss.item()

        if batch_idx % 100 == 0:
            print(batch_idx, len(train_loader), 'Loss: %.3f' % (train_loss))
        if batch_idx % 500 == 0 and batch_idx > 0:
            torch.save(net, 'best-model.pt')
            torch.save(net.state_dict(), 'best-model-parameters.pt')  # official recommended
            print('saved best-model(-parameters).pt')
        if math.isnan(train_loss):
            print('LOSS is nan:', train_loss)
            exit()
            return

        loss.backward()
        optimizer.step()

        #train_loss = loss.item()
        #utils.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f' % (train_loss / (batch_idx + 1)))
        #print(batch_idx, len(train_loader), 'Loss: %.3f' % (train_loss))


start_epoch = 0
for epoch in range(start_epoch, start_epoch + 2000):
    train(epoch)
    scheduler.step()
    torch.save(net, 'best-model.pt')
    torch.save(net.state_dict(), 'best-model-parameters.pt')  # official recommended
