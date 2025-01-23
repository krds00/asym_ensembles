import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.asym_ensembles.modeling.models import MLP, WMLP
import numpy as np
import copy
import concurrent.futures
from tqdm import tqdm
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_mlp_model(args):
    (i, current_seed, in_dim, hidden_dim, out_dim, config, train_loader,
     val_loader, test_loader, criterion, metric_type, dataset_name, rep_i) = args
    
    seed_value = current_seed + i
    set_global_seed(seed_value)

    mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=4, norm=None)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    mlp, train_time, train_losses, val_losses = train_one_model(
        mlp, train_loader, val_loader, criterion, optimizer,
        device=config["device"], max_epochs=config["max_epochs"], patience=config["patience"]
    )
    if config["device"] != "cpu":
        mlp.to("cpu")
    test_metric = evaluate_model(mlp, test_loader, criterion, config["device"], task_type=task_type)

    return (copy.deepcopy(mlp), test_metric)

def train_wmlp_model(args):
    (i, current_seed, in_dim, hidden_dim, out_dim, config["learning_rate"], train_loader,
     val_loader, test_loader, criterion, metric_type, dataset_name, rep_i, mask_params) = args
    
    seed_value_wmlp = current_seed + 2000 + i
    set_global_seed(seed_value_wmlp)

    wmlp = WMLP(in_dim, hidden_dim, out_dim, num_layers=4, mask_params=mask_params, norm=None)
    optimizer_wmlp = torch.optim.AdamW(wmlp.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    wmlp, train_time_wmlp, train_losses_w, val_losses_w = train_one_model(
        wmlp, train_loader, val_loader, criterion, optimizer_wmlp,
        device=config["device"], max_epochs=config["max_epochs"], patience=config["patience"]
    )
    if config["learning_rate"].device != "cpu":
        wmlp.to("cpu")
    test_metric_wmlp = evaluate_model(wmlp, test_loader, criterion, config["device"], task_type=task_type)
    ratio, masked = wmlp.report_masked_ratio()

    return (copy.deepcopy(wmlp), test_metric_wmlp, ratio)
    
def train_one_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    max_epochs=100,
    patience=16,
):
    model.to(device)
    start_time = time.time()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for Xb, yb in train_loader:
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
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

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


def l2_distance_params(model_a, model_b):
    distance = 0.0
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        distance += torch.norm(param_a - param_b, p=2).item() ** 2
    return distance**0.5


def average_pairwise_distance(models):
    n = len(models)
    if n < 2:
        return 0.0
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = l2_distance_params(models[i], models[j])
            distances.append(d)
    return float(np.mean(distances)) if distances else 0.0
