from skimage.transform import resize, rotate
from skimage.filters import gaussian
from skimage.util import random_noise
import torch
import random
import numpy as np

def crop_center(inputs, cropx, cropy, single=False):
	if single:
		y, x = inputs.shape
	else:
		y, x, c = inputs.shape
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)
	return inputs[starty:starty+cropy, startx:startx+cropx, ...]

def rescale(inputs, h, w, single=False):
	if single:
		in_h, in_w = inputs.shape
		if h != in_h or w != in_w:
			inputs = resize(inputs, (h, w))
	else:
		in_h, in_w, c = inputs.shape
		if h != in_h or w != in_w:
			inputs = resize(inputs, (h, w, c))
	return inputs


def rescale_normals(inputs, target, size):
    if not __debug__: print('Rescale: Input, target', inputs.shape, target.shape, size)
    in_h, in_w, _ = inputs.shape
    h, w = size
    if h != in_h or w != in_w:
        inputs = resize(inputs, size, order=1, mode='reflect')
        target = resize(target, size, order=1, mode='reflect')
    return inputs, target


def defined_rotate(inputs, angle):
	inputs = rotate(inputs, angle)
	return inputs


def random_rotate(inputs):
	angle = np.random.randint(-180, 180)
	inputs = rotate(inputs, angle)
	return inputs


def random_crop(inputs, size):
	h, w = inputs.shape[0:2]
	c_h, c_w = size
	if h == c_h and w == c_w:
		return inputs
	x1 = random.randint(0, w - c_w)
	y1 = random.randint(0, h - c_h)
	inputs = inputs[y1: y1 + c_h, x1: x1 + c_w]
	return inputs

def center_crop(inputs, size):
	h, w = inputs.shape[0:2]
	c_h, c_w = size
	if h == c_h and w == c_w:
		return inputs
	# x1 = random.randint(0, w - c_w)
	# y1 = random.randint(0, h - c_h)
	x1 = (w - c_w) // 2
	y1 = (h - c_h) // 2
	inputs = inputs[y1: y1 + c_h, x1: x1 + c_w]
	return inputs


def randomCrop(inputs, target, size, center=True):
	h, w, _ = inputs.shape
	c_h, c_w = size
	if h == c_h and w == c_w:
		return inputs, target
	x1 = random.randint(0, w - c_w)
	y1 = random.randint(0, h - c_h)
	if center:
		x1 = (w - c_w) // 2
		y1 = (h - c_h) // 2
	inputs = inputs[y1: y1 + c_h, x1: x1 + c_w]
	target = target[y1: y1 + c_h, x1: x1 + c_w]
	return inputs, target


def random_noise_aug(inputs, noise_level=0.1):
	return random_noise(inputs, mode="s&p", salt_vs_pepper=noise_level)


def random_rotate_aug(inputs, max_angle=10, specific_angle=None):
	mat = np.eye(3)
	rot_angle = np.random.randint(-max_angle, max_angle)
	if specific_angle is not None:
		rot_angle = specific_angle
	rot = np.deg2rad(rot_angle)
	mat[0:2, 0:2] = [[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]]
	normal_shift = np.clip(inputs @ mat, -1, 1)
	# dist = normal_shift.max() - normal_shift.min()
	# normal_shift = (normal_shift ) / dist
	return normal_shift

def randomNoiseAug(inputs, noise_level=0.05):
	noise = np.random.random(inputs.shape)
	noise = (noise - 0.5) * noise_level
	inputs += noise
	return inputs


def normalToMask(normal, thres=0.01):
	"""
	Due to the numerical precision of uint8, [0, 0, 0] will save as [127, 127, 127] in gt normal,
	When we load the data and rescale normal by N / 255 * 2 - 1, the [127, 127, 127] becomes
	[-0.003927, -0.003927, -0.003927]
	"""
	mask = (np.square(normal).sum(2, keepdims=True) > thres).astype(np.float32)
	return mask


def arrayToTensor(array):
	if array is None:
		return array
	array = np.transpose(array, (2, 0, 1))
	tensor = torch.from_numpy(array)
	return tensor.float()


def random_blur(inputs):
	mode = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
	return gaussian(inputs, sigma=0.1, multichannel=True, mode=random.choice(mode))

