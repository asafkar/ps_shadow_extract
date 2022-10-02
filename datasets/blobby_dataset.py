from __future__ import division
import os
import numpy as np
from imageio import imread
import torchvision

import torch
import torch.utils.data as data
import re
from .transforms import *
import kornia
np.random.seed(0)


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text) ]


def readList(list_path,ignore_head=False, sort=True):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists


class Synth_Dataset(data.Dataset):
	def __init__(self, args, root, split='train', patch_size=None, subset=None):
		self.root = os.path.join(root)
		self.split = split
		self.args = args
		self.subset = subset
		self.shape_list = readList(os.path.join(self.root, split + '_mtrl.txt'))
		self.noise = 0.05
		self.in_img_num = 32  # num of images to use
		self.patch_size = patch_size

	def _getInputPath(self, index):
		shape, mtrl = self.shape_list[index].split('/')
		normal_path = os.path.join(self.root, 'Images', shape, shape + '_normal.png')
		img_dir = os.path.join(self.root, 'Images', self.shape_list[index])
		img_list = readList(os.path.join(img_dir, '%s_%s.txt' % (shape, mtrl)))

		data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
		select_idx = np.random.permutation(data.shape[0])[:self.in_img_num]
		idxs = ['%03d' % (idx) for idx in select_idx]
		data = data[select_idx, :]
		imgs = [os.path.join(img_dir, img) for img in data[:, 0]]
		lights = data[:, 1:4].astype(np.float32)
		return normal_path, imgs, lights

	def __getitem__(self, index):
		normal_path, img_list, lights = self._getInputPath(index)
		normal = imread(normal_path).astype(np.float32) / 255.0 * 2 - 1
		imgs = []
		for i in img_list:
			img = imread(i).astype(np.float32) / 255.0
			imgs.append(img)
		img = np.concatenate(imgs, 2)

		h, w, c = img.shape
		if self.patch_size:
			crop_h, crop_w = self.patch_size, self.patch_size
			sc_h = np.random.randint(crop_h, h)
			sc_w = np.random.randint(crop_w, w)
			img, normal = rescale_normals(img, normal, [sc_h, sc_w])
			img, normal = randomCrop(img, normal, [crop_h, crop_w])
			h, w = crop_h, crop_w

		if self.split == 'train':
			img = (img * np.random.uniform(1, 3)).clip(0, 2)  # color augmentation
			img = randomNoiseAug(img, self.noise)  # noise augmentation

		mask = normalToMask(normal)
		normal = normal * mask.repeat(3, 2)
		norm = np.sqrt((normal * normal).sum(2, keepdims=True))
		normal = normal / (norm + 1e-10)

		item = {'normal_gt': normal, 'imgs': img, 'silhouette': mask}
		for k in item.keys():
			item[k] = arrayToTensor(item[k])

		item['light'] = torch.from_numpy(lights).view(-1, 1, 1).float()

		if self.split == 'train':
			aug1 = kornia.augmentation.ColorJitter(0.2, 0.1, 0.0, 0.0, p=0.5)
			aug2 = kornia.augmentation.ColorJitter(0.0, 0.0, 0.1, 0.1, p=0.5)
			item['imgs'] = aug2(aug1(item['imgs'].reshape(-1, 3, h, w)))
			item['imgs'] = item['imgs'].reshape(-1, h, w)

		return item

	def __len__(self):
		return len(self.shape_list) if not self.subset else self.subset