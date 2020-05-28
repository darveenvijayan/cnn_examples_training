from torch.nn.init import kaiming_normal_
import torch
import torch.optim as optim
from data_preparation import AlexNetDataset
from models import AlexNet
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2
from time import time

def init_weights(model):
	# TODO: change it to more general function
	conv_list = [0, 4, 8, 10, 12]
	fc_list = [1, 4, 6]
	for i in conv_list:
		kaiming_normal_(model.conv_base[i].weight)
	for i in fc_list:
		kaiming_normal_(model.fc_base[i].weight)
	return model

def get_accuracy(loader, model, device, loss_func = torch.nn.CrossEntropyLoss()):
	num_correct = 0
	num_samples = 0
	model.eval()  # set model to evaluation mode
	losses = []
	with torch.no_grad():
		for (imgs, labels) in loader:
			imgs = imgs.to(device = device, dtype = dtype)  # move to device, e.g. GPU
			labels = labels.to(device = device, dtype = torch.long)

			scores = model(imgs)
			loss = loss_func(scores, labels)
			losses.append(loss)

			_, preds = scores.max(1)
			num_correct += (preds == labels).sum()
			num_samples += preds.size(0)
			# if num_samples > 1000: break

		acc = 100 * float(num_correct) / num_samples
		loss = sum(losses) / len(losses)
	return acc, loss


def train_my(loader, model, optimizer, epochs = 3, device = 'cpu', loss_func = torch.nn.CrossEntropyLoss()):
	hist = {
		'train': {
			'accuracy': [], 'loss': []
		},
		'validate': {
			'accuracy': [], 'loss': []
		}
	}

	for epoch in range(epochs):
		t = time()
		print('-' * 16)
		print('--- Epoch {}{} ---'.format(' ' * (2 - len(str(epoch))), epoch))

		tacc, vacc = 0, 0
		tloss, vloss = 0, 0

		for idx, (imgs, labels) in enumerate(loader['train']):
			model.train()  # put model to training mode
			imgs = imgs.to(device = device, dtype = dtype)
			labels = labels.to(device = device, dtype = torch.long)

			scores = model(imgs)
			loss = loss_func(scores, labels)

			# Zero out all of the gradients for the variables which the optimizer will update.
			optimizer.zero_grad()

			# Backwards pass and computing gradients
			loss.backward()
			optimizer.step()

			# checking accuracy
			_, preds = scores.max(1)
			tacc += (preds == labels).sum() / preds.size(0)
			tloss += loss.item()

			temp_acc, temp_loss = get_accuracy(loader['val'], model, device = device, loss_func = loss_func)
			vacc += temp_acc
			vloss += temp_loss

		hist['train']['accuracy'].append(tacc / idx)
		hist['train']['loss'].append(tloss / idx) # TODO: why in Stanford they transfer to numpy?
		hist['validate']['accuracy'].append(vacc / idx)
		hist['validate']['loss'].append(vloss / idx)


		# displaying info
		print('Loss: train = {}, validate = {}'.format(tloss, vloss))
		print('Accuracy: train = {}, validate = {}'.format(round(tacc, 2), vacc))
		t = int(time() - t)
		t_min, t_sec = str(t // 60), str(t % 60)
		print('It took {}{} min. {}{} sec.'.format(' ' * (2 - len(t_min)), t_min, ' ' * (2 - len(t_sec)), t_sec))
		print('-' * 16)
		print()
	print('-' * 16)
	return hist

#
# # A simple train loop that you can use. You can seperate different train and val functions also.
# def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=25):
# 	since = time.time()
#
# 	train_batch_loss = []
# 	train_epoch_loss = []
# 	val_epoch_loss = []
#
# 	for epoch in range(num_epochs):
# 		print('Epoch {}/{}'.format(epoch + 1, num_epochs))
# 		print('-' * 15)
#
# 		# You perform validation test after every epoch
# 		for phase in ['train', 'val']:
# 			if phase == 'train':
# 				model.train()
# 			else:
# 				model.eval()
#
# 			for idx, (inputs, labels) in enumerate(data_loader[phase]):
# 				inputs = inputs.to(device)
# 				labels = labels.to(device)
#
# 				# zero accumulated gradients
# 				optimizer.zero_grad()
#
# 				# During train phase we want to remember history for grads
# 				# and during val we do not want history of grads
# 				with torch.set_grad_enabled(phase == 'train'):
# 					outputs = model(inputs)
# 					loss = criterion(outputs, labels)
#
# 					_, preds = torch.max(outputs, 1)
#
# 					if idx % 200 == 0:
# 						train_batch_loss.append(loss.item())
# 						print('Epoch {}: {}/{} step in progress'.format(epoch + 1, idx, len(data_loader)))
#
# 					if phase == 'train':
# 						loss.backward()
# 						optimizer.step()
#
# 				running_loss += loss.item() * inputs.size(0)
# 				running_corrects += torch.sum(preds == labels.data)
#
# 			epoch_loss = running_loss / len(data_loader[phase].dataset)
# 			epoch_acc = running_corrects.double() / len(data_loader[phase].dataset)
#
# 			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#
# 			if phase == 'val':
# 				val_epoch_loss.append((epoch_loss, epoch_acc))
# 				scheduler.step(loss.item())
# 			else:
# 				train_epoch_loss.append((epoch_loss, epoch_acc))
#
# 		print()
#
# 	time_elapsed = time.time() - since
# 	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#
# 	return model


if __name__ == '__main__':
	USE_GPU = True
	dtype = torch.float32  # TODO: find out how it affects speed and accuracy
	device = torch.device('cuda:0' if USE_GPU and torch.cuda.is_available() else 'cpu')

	model = AlexNet()
	model = model.to(device = device)  # move the model parameters to CPU/GPU

	# initialize weights
	model = init_weights(model)
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
	# optimizer with Nesterov momentum
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

	# create data loader
	data_train = AlexNetDataset(root = os.getcwd() + '/image_net_10', train = True)
	data_val = AlexNetDataset(root = os.getcwd() + '/image_net_10', train = False)
	data_loader = {
		'train': DataLoader(data_train, batch_size = 32, shuffle = True, num_workers = 6),
		'val': DataLoader(data_val, batch_size=32, shuffle = True, num_workers = 6)
	}
	hist = train_my(data_loader, model, optimizer, epochs = 10, device = device)
