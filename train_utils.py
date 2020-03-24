from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np

#imagenet_data = datasets.ImageNet('./data')
#data_loader = DataLoader(imagenet_data,
#			   		     batch_size=16,
#						 shuffle=True)

def loss_func(gen, dis, data, flawed):
	content_loss = nn.MSELoss(reduction='mean')(data, gen(flawed))
	adv_loss = (-torch.log(dis(gen(flawed)))).sum()
	return  content_loss + 1e-4 * adv_loss

def train(gen, dis, optimizer, data_loader, device, epoch):
	for data, flawed in data_loader:
		data.requires_grad_(True)
		flawed.requires_grad_(True)
		data, flawed = data.to(device), flawed.to(device)
		gen.zero_grad()
		dis.zero_grad()
		loss = loss_func(gen, dis, data, flawed)
		print("%d epoch's Loss: %lf" %(epoch + 1, loss))
		loss.backward()
		optimizer.step()

def test(gen, dis, test_data, epoch, device):
	fwd = gen(test_data)
	fwd_tensor = fwd[0]
	save_image(fwd_tensor, '%d.png' %epoch)