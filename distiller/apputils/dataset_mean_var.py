import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

train_path = '/gpfs/group/mtk2/cyberstar/sms821/SpikingNN/quantized_snn/data.tinyimagenet/tiny-imagenet-200/train'

transform = transforms.Compose( [transforms.ToTensor()])
imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
data_loader = torch.utils.data.DataLoader( imagenet_data, batch_size=64, shuffle=False, num_workers=10)

mean = 0.0
for images, _ in data_loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean = mean / len(data_loader.dataset)
print (mean)
np.savetxt('tiny_imagenet_mean.txt', mean)

var = 0.0
h, w = 0, 0
for images, _ in data_loader:
    h, w = images.size(2), images.size(3)
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])

#std = torch.sqrt(var / (len(data_loader.dataset)*64*64))
std = torch.sqrt(var / (len(data_loader.dataset)*h*w))
print (std)
np.savetxt('tiny_imagenet_std.txt', std)
