from models.fpn3d import FPN
import torch


def model_factory():
    first_batch = torch.zeros((4, 3, 3, 480, 640))
    model = FPN(first_batch)
    return model
