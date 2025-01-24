def main(cfg): 
    import sys
    import time
    from pathlib import Path
    import torch
    import torch.nn as nn
    import wandb
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from src.asym_ensembles.data_loaders import load_dataset
    from src.asym_ensembles.modeling.training import (
        set_global_seed,
        train_one_model,
        evaluate_model,
        evaluate_ensemble,
        average_pairwise_distance,
        train_mlp_model,
        train_wmlp_model
    )
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
    from joblib import Parallel, delayed
    import multiprocessing
    num_cpu = multiprocessing.cpu_count()
    n_jobs = max(1, num_cpu)

    wandb.init(
        project="DeepEnsembleProject",
        config=cfg,
        name="Full Experiment",
        settings=wandb.Settings(start_method="fork")
    )
    config = wandb.config
    
    table1 = wandb.Table(
        columns=[
            "dataset_name", "hidden_dim", "repeat_index", "model_index",
            "metric_type", "metric", "masked_ratio", "model_type",
            "train_time", "epochs_until_stop"
        ]
    )
    
    table2 = wandb.Table(
        columns=[
            "dataset_name", "hidden_dim", "repeat_index", "metric_type",
            "avg_dist_mlp", "avg_dist_wmlp",
            "mean_mlp_metric", "std_mlp_metric", "min_mlp_metric", "max_mlp_metric",
            "mean_wmlp_metric", "std_wmlp_metric", "min_wmlp_metric", "max_wmlp_metric",
            "ens_size", "mlp_ens_metric", "wmlp_ens_metric"
        ]
    )
    try:
        for (dataset_name, task_type) in config.all_datasets:
            train_ds, val_ds, test_ds = load_dataset(dataset_name=dataset_name)
            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
            
            for hidden_dim in config.hidden_dims:
                print(f"\nDataset: {dataset_name}, Hidden_dim: {hidden_dim}")
                if hidden_dim in [64, 128]:
                    second_nfix = 3
                else:
                    second_nfix = 4
                mask_params = {
                    0: {'mask_constant': 1, 'mask_type': config.mask_type, 'do_normal_mask': True, 'num_fixed': 2},
                    1: {'mask_constant': 1, 'mask_type': config.mask_type, 'do_normal_mask': True, 'num_fixed': second_nfix},
                    2: {'mask_constant': 1, 'mask_type': config.mask_type, 'do_normal_mask': True, 'num_fixed': second_nfix},
                    3: {'mask_constant': 1, 'mask_type': config.mask_type, 'do_normal_mask': True, 'num_fixed': second_nfix},
                }
                in_dim = train_ds.tensors[0].shape[1]
                if task_type == "regression":
                    out_dim = 1
                    criterion = nn.MSELoss()
                    metric_type = "rmse"
                else:
                    out_dim = len(torch.unique(train_ds.tensors[1]))
                    criterion = nn.CrossEntropyLoss()
                    metric_type = "accuracy"
        
                for rep_i in range(config.repeats):
                    print(f"\nRepetition {rep_i + 1}/{config.repeats}")
                    current_seed = config.base_seed + rep_i * 10000
        
                    mlp_metrics = []
                    wmlp_metrics = []
                    wmlp_masked_ratios = []
                    mlp_models = []
                    wmlp_models = []
        
                    mlp_args = [
                        (
                            i, current_seed, in_dim, hidden_dim, out_dim, copy.deepcopy(cfg),
                            train_loader, val_loader, test_loader, criterion,
                            metric_type, dataset_name, rep_i, task_type
                        )
                        for i in range(cfg["total_models"])
                    ]
                            
                    print("Training MLP models...")
                    mlp_results = Parallel(n_jobs=n_jobs)(
                        delayed(train_mlp_model)(arg) for arg in tqdm(mlp_args, desc="Training MLP")
                    )
                    
                    for model, metric, train_time_val, used_epochs in mlp_results:
                        mlp_models.append(model)
                        mlp_metrics.append(metric)
                        table1.add_data(
                            dataset_name,
                            hidden_dim,
                            rep_i + 1,
                            len(mlp_metrics),
                            metric_type,
                            metric,
                            0,
                            "mlp",
                            train_time_val,
                            used_epochs
                        )
                            
                    wmlp_args = [
                        (
                            i, current_seed, in_dim, hidden_dim, out_dim, copy.deepcopy(cfg),
                            train_loader, val_loader, test_loader, criterion,
                            metric_type, dataset_name, rep_i, mask_params, task_type
                        )
                        for i in range(cfg["total_models"])
                    ]
                                
                    print("Training WMLP models...")
                    wmlp_results = Parallel(n_jobs=n_jobs)(
                        delayed(train_wmlp_model)(arg) for arg in tqdm(wmlp_args, desc="Training WMLP")
                    )
                    
                    for model, metric_wmlp, ratio, train_time_val_w, used_epochs_w in wmlp_results:
                        wmlp_models.append(model)
                        wmlp_metrics.append(metric_wmlp)
                        wmlp_masked_ratios.append(ratio)
                        table1.add_data(
                            dataset_name,
                            hidden_dim,
                            rep_i + 1,
                            len(wmlp_metrics),
                            metric_type,
                            metric_wmlp,
                            ratio,
                            "wmlp",
                            train_time_val_w,
                            used_epochs_w
                        )
        
                    avg_dist_mlp = average_pairwise_distance(mlp_models)
                    avg_dist_wmlp = average_pairwise_distance(wmlp_models)
        
                    avg_wmlp_masked_ratio = float(np.mean(wmlp_masked_ratios)) if wmlp_masked_ratios else 0.0
        
                    mean_mlp_metric = float(np.mean(mlp_metrics))
                    std_mlp_metric = float(np.std(mlp_metrics))
                    min_mlp_metric = float(np.min(mlp_metrics))
                    max_mlp_metric = float(np.max(mlp_metrics))
        
                    mean_wmlp_metric = float(np.mean(wmlp_metrics))
                    std_wmlp_metric = float(np.std(wmlp_metrics))
                    min_wmlp_metric = float(np.min(wmlp_metrics))
                    max_wmlp_metric = float(np.max(wmlp_metrics))
        
                    ensemble_results_mlp = {}
                    ensemble_results_wmlp = {}
                    for ens_size in config.ensemble_sizes:
                        mlp_sub = mlp_models[:ens_size]
                        wmlp_sub = wmlp_models[:ens_size]
                        ens_metric_mlp = evaluate_ensemble(mlp_sub, test_loader, config.device, task_type=task_type)
                        ens_metric_wmlp = evaluate_ensemble(wmlp_sub, test_loader, config.device, task_type=task_type)
                        ensemble_results_mlp[ens_size] = ens_metric_mlp
                        ensemble_results_wmlp[ens_size] = ens_metric_wmlp
        
                    for ens_size in config.ensemble_sizes:
                        table2.add_data(
                            dataset_name,
                            hidden_dim,
                            rep_i + 1,
                            metric_type,
                            avg_dist_mlp,
                            avg_dist_wmlp,
                            mean_mlp_metric,
                            std_mlp_metric,
                            min_mlp_metric,
                            max_mlp_metric,
                            mean_wmlp_metric,
                            std_wmlp_metric,
                            min_wmlp_metric,
                            max_wmlp_metric,
                            ens_size,
                            ensemble_results_mlp[ens_size],
                            ensemble_results_wmlp[ens_size]
                        )
        
                    print(f"Repetition {rep_i + 1}/{config.repeats} finished.")
    
    except Exception as e:
        print(e)
    
    finally:
        wandb.log({
            "Model Metrics": table1,
            "Aggregate Metrics": table2
        })
        
        wandb.finish()


if __name__ == "__main__":
    cfg = {
        "batch_size": 256,
        "max_epochs": 1000,
        "patience": 16,
        "learning_rate": 1e-3,
        "weight_decay": 3e-2,
        "hidden_dims": [64
                        , 128, 256
                       ],
        "ensemble_sizes": [2
                           , 4, 8, 16, 32, 64
                          ],
        "total_models": 64,             # max(ensemble_sizes)
        "repeats": 10,                  # different seeds
        "mask_type": "random_subsets",
        "base_seed": 1234,
        "device": "cpu", # parallel by cpu
        "all_datasets": [
            ["california", "regression"],
            ["otto", "classification"],
            ["telcom", "classification"],
            ["mnist", "classification"]
        ]
    }
    
    main(cfg)
