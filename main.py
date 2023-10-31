import torch
from model_factory import model_factory

model = model_factory()
model = model.cuda()

# batch_size, sequence ( 3 images as 1 sequence ), channel (rgb), height, width
batch_sample = torch.rand((4, 3, 3, 480, 640)).cuda()

model(batch_sample)
