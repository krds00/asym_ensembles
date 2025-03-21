def main(cfg):
    import copy
    import multiprocessing

    import numpy as np
    import torch
    import torch.nn as nn
    from joblib import Parallel, delayed
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import wandb
    from src.asym_ensembles.data_loaders import load_dataset
    from src.asym_ensembles.modeling.models import InterpolatedModel
    from src.asym_ensembles.modeling.training import (
        average_pairwise_distance,
        evaluate_ensemble,
        evaluate_model,
        train_mlp_model,
        train_wmlp_model,
    )

    num_cpu = multiprocessing.cpu_count()
    n_jobs = max(1, num_cpu)

    wandb.init(
        project="DeepEnsembleProject",
        config=cfg,
        name="Full Experiment",
        settings=wandb.Settings(start_method="fork"),
    )
    config = wandb.config

    table1 = wandb.Table(
        columns=[
            "dataset_name",
            "hidden_dim",
            "repeat_index",
            "model_index",
            "metric_type",
            "metric",
            "masked_ratio",
            "model_type",
            "train_time",
            "epochs_until_stop",
        ]
    )

    table2 = wandb.Table(
        columns=[
            "dataset_name",
            "hidden_dim",
            "repeat_index",
            "metric_type",
            "avg_dist_mlp",
            "avg_dist_wmlp",
            "mean_mlp_metric",
            "std_mlp_metric",
            "min_mlp_metric",
            "max_mlp_metric",
            "mean_wmlp_metric",
            "std_wmlp_metric",
            "min_wmlp_metric",
            "max_wmlp_metric",
            "ens_size",
            "mlp_ens_metric",
            "wmlp_ens_metric",
            "mlp_interp_metric",
            "wmlp_interp_metric",
        ]
    )
    try:
        for dataset_name, task_type in config.all_datasets:
            train_ds, val_ds, test_ds = load_dataset(dataset_name=dataset_name)
            train_loader = DataLoader(
                train_ds, batch_size=config.batch_size, shuffle=True
            )
            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
            test_loader = DataLoader(
                test_ds, batch_size=config.batch_size, shuffle=False
            )
            for hidden_dim in config.hidden_dims:  # TODO: where do you specify it?
                print(f"\nDataset: {dataset_name}, Hidden_dim: {hidden_dim}")
                if hidden_dim in [64, 128]:
                    second_nfix = 3
                else:
                    second_nfix = 4
                mask_params = {
                    0: {
                        "mask_constant": 1,
                        "mask_type": config.mask_type,
                        "do_normal_mask": True,
                        "num_fixed": 2,
                    },
                    1: {
                        "mask_constant": 1,
                        "mask_type": config.mask_type,
                        "do_normal_mask": True,
                        "num_fixed": second_nfix,
                    },
                    2: {
                        "mask_constant": 1,
                        "mask_type": config.mask_type,
                        "do_normal_mask": True,
                        "num_fixed": second_nfix,
                    },
                    3: {
                        "mask_constant": 1,
                        "mask_type": config.mask_type,
                        "do_normal_mask": True,
                        "num_fixed": second_nfix,
                    },
                }
                in_features = train_ds.tensors[0].shape[1]
                if task_type == "regression":
                    out_features = 1
                    criterion = nn.MSELoss()
                    metric_type = "rmse"
                else:
                    out_features = len(torch.unique(train_ds.tensors[1]))
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
                            i,
                            current_seed,
                            in_features,
                            hidden_dim,
                            out_features,
                            copy.deepcopy(cfg),
                            train_loader,
                            val_loader,
                            test_loader,
                            criterion,
                            metric_type,
                            dataset_name,
                            rep_i,
                            task_type,
                        )
                        for i in range(cfg["total_models"])
                    ]

                    print("Training MLP models...")
                    mlp_results = Parallel(n_jobs=n_jobs)(
                        delayed(train_mlp_model)(arg)
                        for arg in tqdm(mlp_args, desc="Training MLP")
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
                            used_epochs,
                        )

                    wmlp_args = [
                        (
                            i,
                            current_seed,
                            in_features,
                            hidden_dim,
                            out_features,
                            copy.deepcopy(cfg),
                            train_loader,
                            val_loader,
                            test_loader,
                            criterion,
                            metric_type,
                            dataset_name,
                            rep_i,
                            mask_params,
                            task_type,
                        )
                        for i in range(cfg["total_models"])
                    ]

                    print("Training WMLP models...")
                    wmlp_results = Parallel(n_jobs=n_jobs)(
                        delayed(train_wmlp_model)(arg)
                        for arg in tqdm(wmlp_args, desc="Training WMLP")
                    )

                    for (
                        model,
                        metric_wmlp,
                        ratio,
                        train_time_val_w,
                        used_epochs_w,
                    ) in wmlp_results:
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
                            used_epochs_w,
                        )

                    avg_dist_mlp = average_pairwise_distance(mlp_models)
                    avg_dist_wmlp = average_pairwise_distance(wmlp_models)
                    mean_mlp_metric = float(np.mean(mlp_metrics))
                    std_mlp_metric = float(np.std(mlp_metrics))
                    min_mlp_metric = float(np.min(mlp_metrics))
                    max_mlp_metric = float(np.max(mlp_metrics))

                    mean_wmlp_metric = float(np.mean(wmlp_metrics))
                    std_wmlp_metric = float(np.std(wmlp_metrics))
                    min_wmlp_metric = float(np.min(wmlp_metrics))
                    max_wmlp_metric = float(np.max(wmlp_metrics))
                    for ens_size in config.ensemble_sizes:
                        mlp_sub = mlp_models[:ens_size]
                        wmlp_sub = wmlp_models[:ens_size]
                        ens_metric_mlp = evaluate_ensemble(
                            mlp_sub, test_loader, config.device, task_type=task_type
                        )
                        ens_metric_wmlp = evaluate_ensemble(
                            wmlp_sub, test_loader, config.device, task_type=task_type
                        )
                        interp_mlp = InterpolatedModel(mlp_models[:ens_size])
                        interp_mlp_metric = evaluate_model(
                            interp_mlp,
                            test_loader,
                            criterion,
                            config.device,
                            task_type=task_type,
                        )
                        interp_wmlp = InterpolatedModel(wmlp_models[:ens_size])
                        interp_wmlp_metric = evaluate_model(
                            interp_wmlp,
                            test_loader,
                            criterion,
                            config.device,
                            task_type=task_type,
                        )
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
                            ens_metric_mlp,
                            ens_metric_wmlp,
                            interp_mlp_metric,
                            interp_wmlp_metric,
                        )

                    print(f"Repetition {rep_i + 1}/{config.repeats} finished.")

    except Exception as e:
        print(e)

    finally:
        wandb.log({"Model Metrics": table1, "Aggregate Metrics": table2})

        wandb.finish()


def main_moe(cfg):
    import multiprocessing
    from copy import deepcopy

    from joblib import Parallel, delayed
    from tqdm import tqdm

    import wandb
    from src.asym_ensembles.modeling.training import train_moe_single_combination

    num_cpu = multiprocessing.cpu_count()
    n_jobs = max(1, num_cpu)

    wandb.init(
        project="MoE_Experiments",
        config=cfg,
        name="MoE Experiment",
        settings=wandb.Settings(start_method="fork"),
    )
    config = wandb.config
    table3 = wandb.Table(
        columns=[
            "dataset_name",
            "num_experts",
            "hidden_dim",
            "repeat_index",
            "metric_type",
            "model_type",
            "metric",
            "num_epochs",
            "train_time",
            "gating_type",
        ]
    )

    alpha_table = wandb.Table(
        columns=[
            "dataset_name",
            "num_experts",
            "hidden_dim",
            "repeat_index",
            "model_type",
            "gating_type",
            "alpha_list",  # список средних α за эпохи
        ]
    )

    combos = []
    for dataset_name, task_type in config["all_datasets"]:
        for num_experts in config["num_experts"]:
            for model_type_str in config["model_type_str"]:
                for gating_type in config["gating_type"]:
                    for rep_i in range(config["repeats"]):
                        combos.append(
                            (
                                dataset_name,
                                task_type,
                                num_experts,
                                64,
                                model_type_str,
                                rep_i,
                                gating_type,
                                deepcopy(cfg),
                            )
                        )

    results = Parallel(n_jobs=n_jobs)(
        delayed(train_moe_single_combination)(combo)
        for combo in tqdm(combos, desc="MoE combos")
    )

    for row in results:
        (
            dataset_name,
            num_experts,
            hidden_dim,
            rep_index,
            metric_type,
            model_type_str,
            test_metric,
            num_epochs,
            train_time,
            gating_type,
            alpha_list,
        ) = row
        table3.add_data(
            dataset_name,
            num_experts,
            hidden_dim,
            rep_index,
            metric_type,
            model_type_str,
            test_metric,
            num_epochs,
            train_time,
            gating_type,
        )
        if alpha_list is not None:
            alpha_table.add_data(
                dataset_name,
                num_experts,
                hidden_dim,
                rep_index,
                model_type_str,
                gating_type,
                alpha_list,
            )

    wandb.log({"MoE_Results_Table3": table3, "MoE_Alpha_Table": alpha_table})
    wandb.finish()


# if __name__ == "__main__":
#     cfg = {
#         "batch_size": 256,
#         "max_epochs": 100,
#         "patience": 3,
#         "learning_rate": 1e-3,
#         "weight_decay": 3e-2,
#         "hidden_dims": [
#             64,
#             # 128, 256
#         ],
#         "ensemble_sizes": [
#             2,
#             4,
#             8,
#             # , 16, 32, 64
#         ],
#         "total_models": 8,  # max(ensemble_sizes)
#         "repeats": 1,  # different seeds
#         "mask_type": "random_subsets",
#         "base_seed": 1234,
#         "device": "cpu",  # parallel by cpu
#         "all_datasets": [
#             ["california", "regression"],
#             # ["otto", "classification"],
#             # ["mnist", "classification"],
#             # ["adult", "classification"],
#             # ["churn", "classification"],
#         ],
#     }
#     main(cfg)

if __name__ == "__main__":
    cfg = {
        "batch_size": 256,
        "max_epochs": 1000,
        "patience": 16,
        "learning_rate": 1e-3,
        "weight_decay": 3e-2,
        "repeats": 1,
        "num_experts": [
            2,
            # 4,
            # 8,
            # 16, 32
        ],
        "mask_type": "random_subsets",
        "base_seed": 1234,
        "device": "cpu",
        "all_datasets": [
            # ["california", "regression"],
            ["otto", "classification"],
            # ["mnist", "classification"],
            # ["adult", "classification"],
            ["churn", "classification"],
        ],
        "model_type_str": ["mlp", "wmlp", "imlp", "iwmlp"],
        "gating_type": ["standard", "gumbel"],
    }
    main_moe(cfg)
