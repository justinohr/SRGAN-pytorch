import torch
import torch.nn as nn

class SRGAN_gen(nn.Module):
	def __init__(self, B=5, mode="RGB"):
		super(SRGAN_gen, self).__init__()
		self.B = B
		if mode == "RGB":
			self.ch_in = 3
		else:
			self.ch_in = 1
		self.layers_1 = self._make_layers_1()
		self.layers_2 = self._make_layers_2()
		self.layers_3 = self._make_layers_3()
	
	def pconvBlock(self):
		block = nn.Sequential(
			nn.Conv2d(64,256, kernel_size=3, stride=1, padding=1),
			nn.PixelShuffle(2),
			nn.PReLU(),
			nn.Conv2d(64,256, kernel_size=3,stride=1, padding=1),
			nn.PixelShuffle(2),
			nn.PReLU())
		return block

	def _make_layers_1(self):
		layers = [nn.Conv2d(self.ch_in, 64,
						kernel_size=9, stride=1, padding=4),
				   nn.PReLU()]
		return nn.Sequential(*layers)

	def _make_layers_2(self):
		layers = []
		for i in range(self.B):
			layers += [BottleNeck()]
		layers += [nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)]
		layers += [nn.BatchNorm2d(64)]
		return nn.Sequential(*layers)

	def _make_layers_3(self):
		layers = []
		layers += [self.pconvBlock()]
		layers += [nn.Conv2d(64,self.ch_in, 
					kernel_size=9, stride=1, padding=4)]
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.layers_1(x)
		out = self.layers_2(out) + out
		out = self.layers_3(out)
		return out

class BottleNeck(nn.Module):
	def __init__(self):
		super(BottleNeck, self).__init__()
		self.layers = self._make_layers()

	def _make_layers(self):
		block = nn.Sequential(
			nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.PReLU(),
			nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64))
		return block

	def forward(self, x):
		return self.layers(x) + x

class SRGAN_dis(nn.Module):
	def __init__(self, mode="RGB"):
		super(SRGAN_dis, self).__init__()
		if mode == "RGB":
			self.ch_in = 3
		else:
			self.ch_in = 1
		self.conv = self._make_layers()
		self.fc1 = nn.Linear(512*3*3, 1024)
		self.fc2 = nn.Linear(1024,1)

	def Block(self, ch_in, ch_out, k, s):
		block = nn.Sequential(
			nn.Conv2d(ch_in,ch_out, kernel_size=k, stride=s, padding=0),
			nn.BatchNorm2d(ch_out),
			nn.LeakyReLU(negative_slope=0.2))
		return block

	def _make_layers(self):
		layers = nn.Sequential(
				nn.Conv2d(self.ch_in, 64,
						kernel_size=3, stride=1, padding=0),
				nn.LeakyReLU(),

				self.Block(64,64, 3,2),
				self.Block(64,128, 3,1),

				self.Block(128,128, 3,2),
				self.Block(128,256, 3,1),

				self.Block(256,256, 3,2),
				self.Block(256,512, 3,1),

				self.Block(512,512, 3,2))
		return layers

	def forward(self, x):
		out = self.conv(x)
		out = out.view(-1, self.num_flat_features(out))
		out = nn.LeakyReLU(negative_slope=0.2)(self.fc1(out))
		out = nn.Sigmoid()(self.fc2(out))
		return out
	
	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features