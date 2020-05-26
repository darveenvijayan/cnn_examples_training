from torch import nn
class AlexNet(nn.Module):
	def __init__(self, num_classes=10):
		super(AlexNet, self).__init__()
		self.conv_base = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
			nn.BatchNorm2d(96),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),

			nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),

			nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),

			nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),

			nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.fc_base = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),

			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),

			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.conv_base(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.fc_base(x)
		return x