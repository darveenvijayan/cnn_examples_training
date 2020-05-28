from data_preparation import AlexNetDataset
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2
# data = ImageFolder(root=os.getcwd() + '/raw-img', transform=transform_steps)
if __name__ == '__main__':
	data = AlexNetDataset(root = os.getcwd() + '/image_net_10', train = True)
	data_loader = DataLoader(data, batch_size = 32, shuffle = False, num_workers = 6)

	x, y = next(iter(data_loader))

	# print(x.shape, y.shape)
	# print(x[2])

	sample = x[9].numpy().transpose(1, 2, 0)
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * sample + mean
	inp = np.clip(inp, 0, 1)
	cv2.imshow('image', inp[:, :, ::-1])
	while True:
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"): break
	cv2.destroyAllWindows()

