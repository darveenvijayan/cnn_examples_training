import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class AlexNetDataset(Dataset):
	def __init__(self, root, model = 'AlexNet', train = True):
		self.root = root
		self.model = model
		self.train = train
		# self.transforms = transforms
		classes = os.listdir(root)
		n_class = 10**10

		# check number of photos for each class (to be sure we create balanced set)
		for my_class in classes:
			n = len(os.listdir(os.path.join(root, my_class)))
			if n < n_class: n_class = n
		train_val =  int(0.9 * n_class)

		# create images & labels
		self.imgs = []
		self.labels_idx = []
		self.labels_dict = {}
		for i, my_class in enumerate(classes):
			if train:
				self.imgs.extend([os.path.join(root, my_class, img) for img in  os.listdir(os.path.join(root, my_class))[:train_val]])
				self.labels_idx.extend([i] * train_val)
			else:
				self.imgs.extend(os.listdir(os.path.join(root, my_class))[train_val:n_class])
				self.labels_idx.append([i] * (n_class - train_val))
			self.labels_dict[i] = my_class

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, idx):
		# load images and targets
		img_path = self.imgs[idx]
		img = Image.open(img_path).convert("RGB")
		target = self.labels_idx[idx]

		# TODO: apply augmentation
		if self.model == 'AlexNet':
			if self.train:
				trans = transforms.Compose([
					transforms.Resize(256),
					transforms.RandomCrop(224),
					transforms.RandomHorizontalFlip(0.5),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])
			else:
				trans = transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])
		return trans(img), target

	#
	# def __getitem__(self, index):
	# 	if self.image_weights:
	# 		index = self.indices[index]
	#
	# 	img_path = self.img_files[index]
	# 	label_path = self.label_files[index]
	#
	# 	hyp = self.hyp
	# 	mosaic = True and self.augment  # load 4 images at a time into a mosaic (only during training)
	# 	if mosaic:
	# 		# Load mosaic
	# 		img, labels = load_mosaic(self, index)
	# 		h, w = img.shape[:2]
	# 		ratio, pad = None, None
	#
	# 	else:
	# 		# Load image
	# 		img = load_image(self, index)
	#
	# 		# Letterbox
	# 		h, w = img.shape[:2]
	# 		shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
	# 		img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
	#
	# 		# Load labels
	# 		labels = []
	# 		if os.path.isfile(label_path):
	# 			x = self.labels[index]
	# 			if x is None:  # labels not preloaded
	# 				with open(label_path, 'r') as f:
	# 					x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
	#
	# 			if x.size > 0:
	# 				# Normalized xywh to pixel xyxy format
	# 				labels = x.copy()
	# 				labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
	# 				labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
	# 				labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
	# 				labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
	#
	# 	if self.augment:
	# 		# Augment imagespace
	# 		if not mosaic:
	# 			img, labels = random_affine(img, labels,
	# 										degrees=hyp['degrees'],
	# 										translate=hyp['translate'],
	# 										scale=hyp['scale'],
	# 										shear=hyp['shear'])
	#
	# 		# Augment colorspace
	# 		augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
	#
	# 		# Apply cutouts
	# 		# if random.random() < 0.9:
	# 		#     labels = cutout(img, labels)
	#
	# 	nL = len(labels)  # number of labels
	# 	if nL:
	# 		# convert xyxy to xywh
	# 		labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
	#
	# 		# Normalize coordinates 0 - 1
	# 		labels[:, [2, 4]] /= img.shape[0]  # height
	# 		labels[:, [1, 3]] /= img.shape[1]  # width
	#
	# 	if self.augment:
	# 		# random left-right flip
	# 		lr_flip = True
	# 		if lr_flip and random.random() < 0.5:
	# 			img = np.fliplr(img)
	# 			if nL:
	# 				labels[:, 1] = 1 - labels[:, 1]
	#
	# 		# random up-down flip
	# 		ud_flip = False
	# 		if ud_flip and random.random() < 0.5:
	# 			img = np.flipud(img)
	# 			if nL:
	# 				labels[:, 2] = 1 - labels[:, 2]
	#
	# 	labels_out = torch.zeros((nL, 6))
	# 	if nL:
	# 		labels_out[:, 1:] = torch.from_numpy(labels)
	#
	# 	# Convert
	# 	img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	# 	img = np.ascontiguousarray(img)
	#
	# 	return torch.from_numpy(img), labels_out, img_path, ((h, w), (ratio, pad))
	#
	#
	# def __len__(self):
	# 	return len(self.imgs)