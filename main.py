import srgan
import torch
import torch.nn as nn
import torch.optim as optim
from train_utils import train, test
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os, math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_file = os.listdir('./SR_dataset/291')

train_transform = transforms.Compose([
				transforms.RandomResizedCrop(96),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor()])
BATCH_SIZE = 16

# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def get_gaussian_kernel(kernel_size=3, sigma=1, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=int(kernel_size/2))

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

class TrainDataset(Dataset):
	def __init__(self):
		X = []
		Y = []
		smoothing = get_gaussian_kernel()
		for file in training_file:
			img = Image.open('./SR_dataset/291/' + file)
			img = train_transform(img)

			# applying a gaussian filter
			flawed = smoothing(img.unsqueeze(0))
			# downsampling by a factor of 4
			flawed = flawed[:,:,::4,::4]
			flawed = flawed.squeeze(0)
			
			X.append(img)
			Y.append(flawed)

		self.X = torch.stack(X)
		self.Y = torch.stack(Y)

	def __getitem__(self, index):
		return self.X[index], self.Y[index]

	def __len__(self):
		return len(self.X)

def main():
	generator = srgan.SRGAN_gen().to(device)
	discriminator = srgan.SRGAN_dis().to(device)

	params = list(generator.parameters()) + list(discriminator.parameters())
	optimizer = optim.Adam(params, lr=1e-4)
	trainset = TrainDataset()
	train_loader = DataLoader(dataset=trainset,
							  batch_size=BATCH_SIZE,
							  shuffle=True)
	test_data = Image.open('./SR_dataset/Set5/001_HR.png')
	test_data = transforms.ToTensor()(test_data)
	test_data = test_data.unsqueeze(0)
	test_data = test_data.to(device)
	for epoch in range(10000):
		train(generator, discriminator, optimizer, train_loader, device, epoch)
		if epoch % 1000 == 0:
			test(generator, discriminator, test_data, epoch, device)

if __name__ == "__main__":
	main()