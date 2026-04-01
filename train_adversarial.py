import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import F1Score

import math
import random
import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import time

from dataset import MNISTDataset
from model import Model
from adversarial_utils import pgd_attack


DEVICE = 'cuda:0'
EPOCHS = 100
LR_0 = 0.001
LR_N = 0.0001
BATCH_SIZE = 128

EPS_0 = 0.03
EPS_N = 0.3

eps_schedule = torch.linspace(EPS_0, EPS_N, EPOCHS, device=DEVICE)

train_loss_history = list()
val_loss_history = list()
val_f1_history = list()
val_adv_loss_history = list()
val_adv_f1_history = list()

model = Model().to(DEVICE)
model.print_summary()


train_dataset = MNISTDataset(data_path='./data/mnist/train-00000-of-00001.parquet')
val_dataset = MNISTDataset(data_path='./data/mnist/test-00000-of-00001.parquet')

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss(reduction='sum')
f1_score_fn = F1Score(task="multiclass", num_classes=10).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_0)

warmup_iters = len(train_loader) * EPOCHS * 0.03
regular_iters = len(train_loader) * EPOCHS * 0.97
gamma = math.exp(1 / regular_iters * math.log(LR_N / LR_0))

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1, total_iters=warmup_iters)
regular_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, regular_scheduler], [warmup_iters])

for epoch in range(EPOCHS):
    eps = eps_schedule[epoch]

    model.train()

    train_loss = 0
    val_loss = 0
    val_f1 = 0
    val_adv_loss = 0
    val_adv_f1 = 0

    start = time.time()

    for x, y in train_loader:
        optimizer.zero_grad()

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_adv = torch.as_tensor([random.choice(list(set(range(10)) - set([y[i].item()]))) for i in range(y.shape[0])], device=DEVICE)

        assert torch.all(y != y_adv)

        x_adv = pgd_attack(x, y_adv, model, eps)

        y_pred = model(x_adv)

        loss = criterion(y_pred, y)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()

    model.eval()
    val_y_target = list()
    val_y_pred = list()
    val_y_adv_pred = list()

    for batch_idx, (x, y) in enumerate(val_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.no_grad():
            y_pred = model(x)

        loss = criterion(y_pred, y)

        val_loss += loss.item()
        val_y_pred.append(y_pred.argmax(-1))
        val_y_target.append(y)

        if batch_idx == 0 and (epoch + 1) % 10 == 0:
            grad_vis_save_dir = f'./adversarially_trained_model/grad_vis/epoch{epoch+1}'

            if os.path.exists(grad_vis_save_dir):
                shutil.rmtree(grad_vis_save_dir)
            os.makedirs(grad_vis_save_dir)

            x_grad = x.detach().clone().requires_grad_().to(model.device)
            model.requires_grad_(False)
            y_pred = model(x_grad)
            loss = criterion(y_pred, y_pred.argmax(-1)) # in inference mode we know nothing about real label
            loss.backward()

            grad = x_grad.grad
            grad = grad.cpu().numpy()[:, 0]

            x_np = ((x.detach() + 1) * 127.5).cpu().numpy().astype(np.uint8)[:, 0]
            for i in range(y.shape[0]):
                r = np.abs(grad[i]).max()

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                axes[0].imshow(x_np[i], cmap='gray')
                axes[0].set_title('Original image')
                axes[1].matshow(grad[i], cmap='bwr', vmin=-r, vmax=r)
                axes[1].set_title('Gradient w.r.t. input data')

                fig.savefig(os.path.join(grad_vis_save_dir, f'{i}_{y[i]}.png'))
                plt.close(fig)

            model.requires_grad_()

        if batch_idx == len(val_loader) - 1 and (epoch + 1) % 10 == 0:
            adv_samples_save_dir = f'./adversarially_trained_model/adversarial_samples/epoch{epoch+1}'

            if os.path.exists(adv_samples_save_dir):
                shutil.rmtree(adv_samples_save_dir)
            os.makedirs(adv_samples_save_dir)

            y_adv = torch.as_tensor([random.choice(list(set(range(10)) - set([y[i].item()]))) for i in range(y.shape[0])], device=DEVICE)
            x_adv_vis = pgd_attack(x[:16], y_adv[:16], model, 0.5)

            x_np = ((x.detach() + 1) * 127.5).cpu().numpy().astype(np.uint8)[:, 0]
            x_adv_vis = ((x_adv_vis.detach() + 1) * 127.5).cpu().numpy().astype(np.uint8)[:, 0]

            for i, (x_np_, x_adv_vis_) in enumerate(zip(x_np, x_adv_vis)):
                cv2.imwrite(os.path.join(adv_samples_save_dir, f'{y[i]}_{y_adv[i]}.png'), np.hstack([x_np_, x_adv_vis_]))

    for x, y in val_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_adv = torch.as_tensor([random.choice(list(set(range(10)) - set([y[i].item()]))) for i in range(y.shape[0])], device=DEVICE)
        x_adv = pgd_attack(x, y_adv, model, eps)

        with torch.no_grad():
            y_pred = model(x_adv)

        loss = criterion(y_pred, y)

        val_adv_loss += loss.item()
        val_y_adv_pred.append(y_pred.argmax(-1))

    val_y_pred = torch.cat(val_y_pred)
    val_y_target = torch.cat(val_y_target)
    val_y_adv_pred = torch.cat(val_y_adv_pred)

    train_loss_history.append(train_loss / len(train_dataset))
    val_loss_history.append(val_loss / len(val_dataset))
    val_f1_history.append(f1_score_fn(val_y_pred, val_y_target).item())
    val_adv_loss_history.append(val_adv_loss / len(val_dataset))
    val_adv_f1_history.append(f1_score_fn(val_y_adv_pred, val_y_target).item())

    print(f'Epoch {epoch+1}/{EPOCHS}, lr: {"{:0.2e}".format(lr_scheduler.get_last_lr()[0])}, epoch time: {round(time.time() - start, 2)}.', end='')
    print(f' Train loss: {round(train_loss_history[-1], 6)},', end='')
    print(f' val loss: {round(val_loss_history[-1], 6)}, val f1: {round(val_f1_history[-1], 6)},', end='')
    print(f' val adv loss: {round(val_adv_loss_history[-1], 6)}, val adv f1: {round(val_adv_f1_history[-1], 6)}')
