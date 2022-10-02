import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import torch
import numpy as np
import torch.utils.data as data
from .transforms import *
import kornia


np.random.seed(0)

class BlenderDataset(data.Dataset):
	def __init__(self, args, root, split='train', num_val=2, normalize_lights=True, in_img_num=32, patch_size=None):
		self.num_val = num_val
		self.root = os.path.join(root)
		self.split = split
		self.args = args
		self.normalize_lights = normalize_lights
		self.objects = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
		self.in_img_num = in_img_num
		self.patch_size = patch_size
		self.noise = 0.05
		self.mean = np.asarray([0.0679, 0.0679, 0.0679])
		self.std = np.asarray([0.0413, 0.0413, 0.0413])

	@staticmethod
	def _read_list(list_path):
		with open(list_path) as f:
			lists = f.read().splitlines()
		return lists

	# each index will choose a folder of object_view
	def __getitem__(self, index):
		if self.split == 'test':
			index -= self.num_val  #FIXME!

		light_dir = {}
		obj_name = self.objects[index]
		lines = self._read_list(os.path.join(self.root, obj_name, 'all_object_lights.txt'))
		num_imgs = len(lines)
		lights = torch.empty((num_imgs, 3))

		for idx, line in enumerate(lines):
			name, x, y, z = line.split()
			light_dir[name] = np.asarray((x, y, z), dtype=float)

			if self.normalize_lights:
				light_dir[name] /= np.linalg.norm(light_dir[name])

			lights[idx] = torch.from_numpy(light_dir[name])

		dummy_img_path = os.path.join(self.root, obj_name, [x for x in light_dir.keys()][0] + "_img.png")
		dummy_img = cv2.imread(dummy_img_path)
		h, w, c = dummy_img.shape  # original size

		imgs = []
		shadows = np.empty((num_imgs, h, w))
		for idx, line in enumerate(lines):
			name, x, y, z = line.split()
			img_fname = os.path.join(self.root, obj_name, name + "_shadow1.png")
			shadow = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
			shadow = np.transpose(shadow, (2, 0, 1)).mean(axis=0)
			shadows[idx] = shadow

			img_fname = os.path.join(self.root, obj_name, name + "_img.png")
			img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
			# img -= self.mean
			# img /= np.asarray(self.std)
			imgs.append(img)

		imgs = np.concatenate(imgs, 2)

		path_base = dummy_img_path.split("_0_0")[0]
		normal_gt_path = path_base + "_normal1.png"
		normal = (cv2.cvtColor(cv2.imread(normal_gt_path), cv2.COLOR_BGR2RGB).astype(
			np.float32) / 255.0) * 2 - 1

		# diffuse_color_path = path_base + "_diffuse_color1.png"
		# diffuse_color_gt = cv2.cvtColor(cv2.imread(diffuse_color_path), cv2.COLOR_BGR2RGB).astype(
		# 	np.float32) / 255.0
		depth_exr = cv2.imread(path_base + "_depth1.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]

		depth_img = depth_exr

		if self.patch_size:
			crop_h, crop_w = self.patch_size, self.patch_size
			h, w, _ = img.shape
			if not (h == crop_h and w == crop_w):
				x1 = random.randint(0, w - crop_w)
				y1 = random.randint(0, h - crop_h)
				imgs = imgs[y1: y1 + crop_h, x1: x1 + crop_w]
				normal = normal[y1: y1 + crop_h, x1: x1 + crop_w]
				# diffuse_color_gt = diffuse_color_gt[y1: y1 + crop_h, x1: x1 + crop_w]
				# depth_img = depth_img[y1: y1 + crop_h, x1: x1 + crop_w]
				shadows = shadows[:, y1: y1 + crop_h, x1: x1 + crop_w]

		# depth_img = depth_img - depth_img.min()
		# depth_img /= depth_img.max()
		depth_exr_normed = depth_img

		# silhouette_gt = cv2.imread(path_base + "_silhouette1.png")[:, :, 0] / 255.0
		# silhouette_gt = torch.from_numpy(silhouette_gt).float().to(dev)

		if self.split == 'train':
			imgs = (imgs * np.random.uniform(1, 3)).clip(0, 2)  # color augmentation
			imgs = randomNoiseAug(imgs, self.noise)  # noise augmentation

		mask = normalToMask(normal)
		normal = normal * mask.repeat(3, 2)
		norm = np.sqrt((normal * normal).sum(2, keepdims=True))
		normal = normal / (norm + 1e-10)

		select_idx = np.random.permutation(num_imgs)[:self.in_img_num]

		item = {'normal_gt': normal, 'silhouette': mask,  'imgs': imgs}  # 'diffuse': diffuse_color_gt,
		for k in item.keys():
			item[k] = arrayToTensor(item[k])

		item['light'] = lights.float()[select_idx]
		# item['depth'] = torch.from_numpy(depth_img).float()
		item['shadows'] = torch.from_numpy(shadows).float()[select_idx]
		if self.patch_size:
			item['imgs'] = item['imgs'].reshape(num_imgs,3, self.patch_size, self.patch_size)[select_idx].reshape(-1, self.patch_size, self.patch_size)
		else:
			item['imgs'] = item['imgs'].reshape(num_imgs,3, w, h)[select_idx].reshape(-1, w, h)

		if self.split == 'train':
			aug1 = kornia.augmentation.ColorJitter(0.1, 0.1, 0.0, 0.0, p=0.5)
			aug2 = kornia.augmentation.ColorJitter(0.0, 0.0, 0.1, 0.1, p=0.5)
			item['imgs'] = aug2(aug1(item['imgs'].reshape(-1, 3, crop_h, crop_w)))
			item['imgs'] = item['imgs'].reshape(-1, crop_h,crop_w)

		return item

	def __len__(self):
		# FIXME
		if self.split == 'train':
			return len(self.objects) - self.num_val  # this will cause each epoch to go through all images in this category
		else:
			return self.num_val