import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.asym_ensembles.data_loaders import load_dataset
from src.asym_ensembles.modeling.models import MLP, WMLP
from src.asym_ensembles.modeling.moe import MoE
from src.asym_ensembles.modeling.moie import MoIE


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_mlp_model(args):
    (
        i,
        current_seed,
        in_features,
        hidden_dim,
        out_features,
        config,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        metric_type,
        dataset_name,
        rep_i,
        task_type,
    ) = args

    seed_value = current_seed + i
    set_global_seed(seed_value)

    mlp = MLP(in_features, hidden_dim, out_features, num_layers=4, norm=None)
    optimizer = torch.optim.AdamW(
        mlp.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    mlp, train_time, train_losses, val_losses = train_one_model(
        mlp,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device=config["device"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
    )
    if config["device"] != "cpu":
        mlp.to("cpu")
    test_metric = evaluate_model(
        mlp, test_loader, criterion, config["device"], task_type=task_type
    )

    return (copy.deepcopy(mlp), test_metric, train_time, len(train_losses))


def train_wmlp_model(args):
    (
        i,
        current_seed,
        in_features,
        hidden_dim,
        out_features,
        config,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        metric_type,
        dataset_name,
        rep_i,
        mask_params,
        task_type,
    ) = args

    seed_value_wmlp = current_seed + 2000 + i  # TODO: тут мб одну инициалищацию?
    set_global_seed(seed_value_wmlp)

    wmlp = WMLP(
        in_features, hidden_dim, out_features, num_layers=4, mask_params=mask_params, norm=None
    )
    optimizer_wmlp = torch.optim.AdamW(
        wmlp.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    wmlp, train_time_wmlp, train_losses_w, val_losses_w = train_one_model(
        wmlp,
        train_loader,
        val_loader,
        criterion,
        optimizer_wmlp,
        device=config["device"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
    )
    if config["device"] != "cpu":
        wmlp.to("cpu")
    test_metric_wmlp = evaluate_model(
        wmlp, test_loader, criterion, config["device"], task_type=task_type
    )
    ratio, masked = wmlp.report_masked_ratio()

    return (
        copy.deepcopy(wmlp),
        test_metric_wmlp,
        ratio,
        train_time_wmlp,
        len(train_losses_w),
    )


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
    import wandb

    model.to(device)
    start_time = time.time()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    wait = 0
    alpha_list = [] if isinstance(model, MoE) else None

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0

        if isinstance(model, MoE):
            epoch_alpha_sum = None
            epoch_samples = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * Xb.size(0)
            if isinstance(model, MoE):
                with torch.no_grad():
                    # logits = model.gating_layer(Xb)
                    # alpha = torch.softmax(logits, dim=-1)
                    alpha = model.gate(Xb) if model.gating_type == 'standard' else model.gate(Xb,1)
                    if alpha.dim() == 3:  # for Gumbel Sampling
                        alpha = alpha.mean(dim=0)
                    batch_sum = alpha.sum(dim=0)
                    if epoch_alpha_sum is None:
                        epoch_alpha_sum = batch_sum
                    else:
                        epoch_alpha_sum += batch_sum
                    epoch_samples += Xb.size(0)

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

        if isinstance(model, MoE) and epoch_samples > 0:
            avg_alpha = epoch_alpha_sum / float(epoch_samples)  # (num_experts,)
            alpha_list.append(avg_alpha.cpu().numpy().tolist())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    train_time = time.time() - start_time
    if isinstance(model, MoE) or isinstance(model, MoIE):
        return model, train_time, train_losses, val_losses, alpha_list
    else:
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
        rmse = float(avg_loss ** 0.5)
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
        )  # (num_models, batch_size, out_features)
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
    return distance ** 0.5


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


def train_moe_single_combination(args):
    (
        dataset_name,
        task_type,
        num_experts,
        hidden_dim,
        model_type_str,
        rep_i,
        config,
    ) = args
    current_seed = config["base_seed"] + rep_i * 10000
    set_global_seed(current_seed)
    print(f"gating type:{config['gating_type']}")

    train_ds, val_ds, test_ds = load_dataset(dataset_name)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    if task_type == "regression":
        out_features = 1
        criterion = nn.MSELoss()
        metric_type = "rmse"
    else:
        out_features = len(torch.unique(train_ds.tensors[1]))
        criterion = nn.CrossEntropyLoss()
        metric_type = "accuracy"

    in_features = train_ds.tensors[0].shape[1]
    if model_type_str == "mlp":
        ExpertClass = MLP
        exp_params = {"num_layers": 4, "hidden_dim": hidden_dim}
    else:
        ExpertClass = WMLP

        if hidden_dim in [64, 128]:
            second_nfix = 3
        else:
            second_nfix = 4
        mask_params = {
            0: {
                "mask_constant": 1,
                "mask_type": config["mask_type"],
                "do_normal_mask": True,
                "num_fixed": 2,
            },
            1: {
                "mask_constant": 1,
                "mask_type": config["mask_type"],
                "do_normal_mask": True,
                "num_fixed": second_nfix,
            },
            2: {
                "mask_constant": 1,
                "mask_type": config["mask_type"],
                "do_normal_mask": True,
                "num_fixed": second_nfix,
            },
            3: {
                "mask_constant": 1,
                "mask_type": config["mask_type"],
                "do_normal_mask": True,
                "num_fixed": second_nfix,
            },
        }
        exp_params = {
            "num_layers": 4,  # TODO: hardcode, nice to have it in the config, why here?
            "hidden_dim": hidden_dim,
            "mask_params": mask_params,
        }
    if model_type_str not in ['imlp', 'iwmlp']:
        moe_model = MoE(
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            expert_class=ExpertClass,
            expert_params=exp_params,
            gating_type=config['gating_type']
        )
    else:
        moe_model = MoIE(
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            mask_params=mask_params if model_type_str == 'iwmlp' else None,
            num_layers=4,
            experts_type_str=model_type_str,
            gating_type=config['gating_type']
        )

    optimizer = torch.optim.AdamW(
        moe_model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    moe_model, train_time, train_losses, val_losses, alpha_list = train_one_model(
        moe_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device=config["device"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
    )

    test_metric = evaluate_model(
        moe_model, test_loader, criterion, config["device"], task_type=task_type
    )

    return (
        dataset_name,
        num_experts,
        hidden_dim,
        rep_i + 1,
        metric_type,
        model_type_str,
        test_metric,
        len(train_losses),
        train_time,
        alpha_list,
    )
