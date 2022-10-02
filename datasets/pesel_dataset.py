import os
import cv2
import torch
import numpy as np
import torch.utils.data as data
from datasets import transforms
import glob
np.random.seed(0)


# 23 Illuminations, 1 object
class PESELDataset(data.Dataset):
	def __init__(self, args, root, split='train'):
		args.validate_results = False  # no GT normals here..
		self.root = os.path.join(root)
		self.split = split
		self.args = args
		self.images = None
		self.light_dir = []
		self.names = []
		self.object_types = ["0"]  # fixme - needed to init z size

		if os.path.exists(os.path.join(self.root, 'dirTab.txt')):
			self.info = self._read_list(os.path.join(self.root, 'dirTab.txt'))
			for line in self.info:
				name, x, y = line.split("\t")
				self.names.append(name)
				z = np.sqrt(1 - float(x)**2 - float(y)**2)
				self.light_dir.append((float(x), 1-float(y), z))

			dummy_img_path = os.path.join(self.root, self.names[0])
		else:  # not pesel, other dataset
			for file in glob.glob(os.path.join(self.root) + "/*.jpeg"):
				fn = file.split("\\")[-1] #.split(".jpeg")[0]
				self.names.append(fn)
				self.light_dir.append((0, 0, 0))

			dummy_img_path = os.path.join(self.root, self.names[-1])
		dummy_img = cv2.imread(dummy_img_path)
		self.h, self.w, self.c = dummy_img.shape  # original size

		if args.rescale:
			args.img_h = args.scale_h
			args.img_w = args.scale_w
		else:
			args.img_h = self.h
			args.img_w = self.w

	@staticmethod
	def _read_list(list_path):
		with open(list_path) as f:
			lists = f.read().splitlines()
		return lists

	def __getitem__(self, index):  # chooses a random category

		select_indices = np.random.permutation(len(self.names))[:self.args.input_concat_size]
		img_paths = [os.path.join(self.root, self.names[ii]) for ii in select_indices]
		img = np.empty((self.h, self.w, self.c*self.args.input_concat_size))

		light_dirs = [self.light_dir[ii] for ii in select_indices]
		light_dirs = np.asarray(light_dirs, dtype=float)

		for idx, img_name in enumerate(img_paths):
			img_ = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
			img[:, :, 3 * idx:3 * (idx + 1)] = img_

		crop_h, crop_w = self.args.crop_h, self.args.crop_w

		if self.args.crop:
			img = transforms.center_crop(img, [crop_h, crop_w])

		if self.args.rescale:
			sc_h = self.args.scale_h
			sc_w = self.args.scale_w
			img = transforms.rescale(img, sc_h, sc_w)

		if self.args.color_aug:
			img = (img * np.random.uniform(1, 3)).clip(0, 2)

		if self.args.noise_aug:
			img = transforms.random_noise_aug(img, self.args.noise)

		if self.args.patch_noise:
			img = transforms.add_patch(img)

		# format img as (c,w,h) for torch
		if os.path.exists(os.path.join(self.root, 'silhouette.jpg')):
			silhouette = cv2.imread(os.path.join(self.root, 'silhouette.jpg'))
			silhouette = silhouette[:, :, 0] / 255.0
		else:
			silhouette = np.ones_like(img)[:,:,0]

		if self.args.rescale:
			sc_h = self.args.scale_h
			sc_w = self.args.scale_w
			silhouette = transforms.rescale(silhouette, sc_h, sc_w, single=True)

		dummy_intensities = np.asarray([1, 1, 1])[np.newaxis]

		item = {'img': np.transpose(img, (2, 0, 1)), 'light_dir': light_dirs,
			'mask': silhouette, 'light_intense': dummy_intensities}

		for key in item.keys():  # to torch, to float
			if key == "object": continue
			item[key] = torch.from_numpy(item[key]).float()
			if key == "light_dir" or key == 'light_intense':
				item[key] = item[key].squeeze()

		return item

	def __len__(self):
		return len(self.names)  # this will cause each epoch to go through all images in this category
