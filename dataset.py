import pandas as pd
import numpy as np
import cv2

from torch.utils.data import Dataset
from torch import nn
import torch


class MNISTDataset(Dataset):
    def __init__(self, data_path='./data/mnist/train-00000-of-00001.parquet'):
        data = pd.read_parquet(data_path)
        assert len(data.image) == len(data.label)

        self.images = [cv2.imdecode(np.frombuffer(e['bytes'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE) 
                       for e in data.image]
        self.labels = data.label

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.astype(np.float32) / 127.5 - 1
        image = torch.from_numpy(image).unsqueeze(0)

        return image, self.labels[idx]
    

class AdversarialDataset(Dataset):
    def __init__(self, regular_dataset: MNISTDataset, model: nn.Module, eps: float, loss_fn=nn.CrossEntropyLoss()):
        self.regular_dataset = regular_dataset
        self.model = model
        self.loss_fn = loss_fn

        self.__eps = eps

    @property
    def eps(self):
        return self.__eps
    
    @eps.setter
    def eps(self, value):
        self.__eps = value

    def _pgd_attack(self, x: torch.Tensor, y_target: torch.Tensor,pgd_iterations=5):
        x_grad = x.detach().clone().requires_grad_().to(self.model.device)

        for _ in range(pgd_iterations):
            x_grad.grad = None
            y_pred = self.model.requires_grad_(False)(x_grad)

            self.loss_fn(y_pred, y_target).backward()
            x_grad = torch.clamp(x_grad + self.eps * torch.sign(x_grad.grad), -1, 1)

        return x_grad.detach()
    
    def _fgsm_attack(self, x: torch.Tensor, y_target: torch.Tensor):
        x_grad = x.detach().clone().requires_grad_().to(self.model.device)

        y_pred = self.model.requires_grad_(False)(x_grad)
        self.loss_fn(y_pred, y_target).backward()
        x_grad = torch.clamp(x_grad + self.eps * torch.sign(x_grad.grad), -1, 1)

        return x_grad.detach()
    
    def __getitem__(self, idx):
        x, y = self.regular_dataset[idx]

        x_adv = self._pgd_attack(x, torch.randint(0, 10, (1,)))

        return x_adv, y


    def __len__(self):
        return len(self.regular_dataset)
