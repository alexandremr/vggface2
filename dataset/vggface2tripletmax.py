#!/usr/bin/env python

import os

import numpy as np
import PIL.Image
import torch
from torch.utils import data
import torchvision.transforms
import random
import torch.nn.functional as F
from tqdm import tqdm


class ImgTriplet:
    def __init__(self, anchor, positive, negative):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative


class VGGFaces2(data.Dataset):
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt

    def __init__(self, root, image_list_file, id_label_dict, net, split='train', transform=True, upper=None):
        """
        :param root: dataset directory
        :param image_list_file: contains image file names under root
        :param id_label_dict: X[class_id] -> label
        :param split: train or valid
        :param transform: 
        :param upper: max number of image used for debug
        """
        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.split = split
        self._transform = transform
        self.id_label_dict = id_label_dict
        self.net = net

        self.img_data = []
        img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_file in enumerate(f):
                if not os.path.isfile(self.root + img_file.strip()):
                    #    print('not found', self.root + img_file)
                    continue
                img_file = img_file.strip()  # e.g. n004332/0317_01.jpg
                class_id = img_file.split("/")[0]  # like n004332
                label = self.id_label_dict[class_id]
                img_info.append({
                    'cid': class_id,
                    'img': img_file,
                    'lbl': label,
                })
                if i % 500000 == 0:
                    print("processing: {} images for {}".format(i, self.split))
                if upper and i == upper - 1:  # for debug purpose
                    break
        self.generate_triplets(self.net, img_info)

    def generate_triplets(self, net, img_info):
        counter = 0
        for anchor_info in tqdm(img_info):
        # for anchor_info in self.img_info:
            anchor_label = anchor_info['lbl']
            anchor_img = self.get_img(anchor_info['img'], aug=False)
            anchor_img = self.transform(anchor_img).to('cuda')
            anchor_emb = net.forward_once(anchor_img.unsqueeze(0))

            positives = []
            negatives = []
            for i in range(2):
                label, img_file, _ = self.get_pair(anchor_label, img_info, positive=True)
                pos_img = self.get_img(img_file, aug=False)
                pos_img = self.transform(pos_img).to('cuda')
                pos_emb = net.forward_once(pos_img.unsqueeze(0))
                distance = 1 - F.cosine_similarity(anchor_emb, pos_emb, dim=1)
                positives.append([distance, label, img_file])
            for i in range(10):
                label, img_file, _ = self.get_pair(anchor_label, img_info, positive=False)
                neg_img = self.get_img(img_file, aug=False)
                neg_img = self.transform(neg_img).to('cuda')
                neg_emb = net.forward_once(neg_img.unsqueeze(0))
                distance = 1 - F.cosine_similarity(anchor_emb, neg_emb, dim=1)
                negatives.append([distance, label, img_file])

            positive = max(positives, key=lambda x: x[0])
            negative = min(negatives, key=lambda x: x[0])

            self.img_data.append({
                'anchor_img': img_file,
                'anchor_lbl': label,
                'positive_img': positive[2],
                'positive_lbl': positive[1],
                'negative_img': negative[2],
                'negative_lbl': negative[1],
            })

            counter += 1
            if counter > 40:
                break

    def get_img(self, img_file, aug=True):
        img = PIL.Image.open(os.path.join(self.root, img_file))
        img = torchvision.transforms.Resize((224, 224))(img)
        if self.split == 'train' and aug:
            img = torchvision.transforms.RandomGrayscale(p=0.2)(img)
            img = torchvision.transforms.RandomHorizontalFlip(p=0.5)(img)
            img = torchvision.transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.075)(img)
            img = torchvision.transforms.GaussianBlur(kernel_size=11, sigma=(0.01, 1.15))(img)
            img = torchvision.transforms.RandomRotation(degrees=10)(img)
        else:
            img = torchvision.transforms.CenterCrop(224)(img)

        img = np.array(img, dtype=np.uint8)
        assert len(img.shape) == 3  # assumes color images and no alpha channel

        return img

    def __len__(self):
        return len(self.img_data)

    def get_pair(self, label, img_info, positive=True):
        info = random.choice(img_info)
        img_label = info['lbl']
        if positive:
            while label != img_label:
                info = random.choice(img_info)
                img_label = info['lbl']
        else:
            while label == img_label:
                info = random.choice(img_info)
                img_label = info['lbl']

        label = info["lbl"]
        img_file = info["img"]
        class_id = info["cid"]
        return label, img_file, class_id

    def __getitem__(self, index):
        data = self.img_data[index]
        anchor_label = data['anchor_lbl']
        anchor_img = self.get_img(data['anchor_img'])

        positive_lbl = data['positive_lbl']
        positive_img = self.get_img(data['positive_img'])

        negative_lbl = data['negative_lbl']
        negative_img = self.get_img(data['negative_img'])

        if self._transform:
            anchor = self.transform(anchor_img)
            positive = self.transform(positive_img)
            negative = self.transform(negative_img)
        else:
            anchor = anchor_img
            positive = positive_img
            negative = negative_img

        return anchor, positive, negative

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
