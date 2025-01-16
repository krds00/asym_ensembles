import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_model(
    model, train_loader, val_loader, criterion, optimizer, device, epochs=10
):
    model.to(device)
    start_time = time.time()

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        for i, (Xb, yb) in enumerate(train_loader):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * Xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (Xb, yb) in enumerate(val_loader):
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

    train_time = time.time() - start_time
    return model, train_time, train_losses, val_losses


def evaluate_model(model, test_loader, criterion, device, task_type="regression"):
    model.eval()
    model.to(device)
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (Xb, yb) in enumerate(test_loader):
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            if task_type == "regression":
                # MSE -> RMSE
                loss = criterion(preds, yb)
                total_loss += loss.item() * Xb.size(0)
            else:
                # CrossEntropy
                loss = criterion(preds, yb)
                total_loss += loss.item() * Xb.size(0)
                pred_classes = preds.argmax(dim=1)
                correct += (pred_classes == yb).sum().item()
            total_samples += Xb.size(0)

    avg_loss = total_loss / total_samples
    if task_type == "regression":
        rmse = float(avg_loss**0.5)
        return rmse
    else:
        accuracy = correct / total_samples
        return accuracy


def ensemble_predict(models, X, device, task_type="regression"):
    with torch.no_grad():
        preds_list = []
        for m in models:
            m.eval()
            m.to(device)
            preds = m(X.to(device))
            preds_list.append(preds.cpu())
        stack_preds = torch.stack(
            preds_list, dim=0
        )  # (num_models, batch_size, out_dim)
        if task_type == "regression":
            return stack_preds.mean(dim=0)
        else:
            avg_logits = stack_preds.mean(dim=0)
            return avg_logits.argmax(dim=1)


def evaluate_ensemble(models, test_loader, device, task_type="regression"):
    all_preds = []
    all_targets = []
    for i, (Xb, yb) in enumerate(test_loader):
        yb = yb.to(device)
        ensemble_out = ensemble_predict(models, Xb, device, task_type=task_type)
        all_preds.append(ensemble_out)
        all_targets.append(yb)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if task_type == "regression":
        mse = F.mse_loss(all_preds, all_targets)
        return float(mse.sqrt().item())  # RMSE
    else:
        correct = (all_preds == all_targets).sum().item()
        return correct / len(all_targets)
