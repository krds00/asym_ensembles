import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset


def load_california_housing(test_ratio=0.2, val_ratio=0.1, seed=42):
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=seed + 1
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(-1)
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(-1)
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float().unsqueeze(-1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    return train_ds, val_ds, test_ds


def load_wine_quality(test_ratio=0.2, val_ratio=0.1, seed=42):
    data = load_wine()
    X, y = data.data, data.target

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=seed + 1
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    return train_ds, val_ds, test_ds


def load_covertype_dataset(test_ratio=0.2, val_ratio=0.1, seed=42):
    data = fetch_covtype()
    X, y = data.data, data.target
    y = y - 1  # class range from 1 to 7
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=seed + 1
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()  # Классы → long
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    return train_ds, val_ds, test_ds


def load_otto_csv(
    path_csv="data/raw/otto.csv", test_ratio=0.2, val_ratio=0.1, seed=42
):
    import os

    print(os.getcwd())
    df = pd.read_csv(path_csv)
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    y = df["target"].values
    X = df.drop(columns=["target"]).values

    if isinstance(y[0], str) and y[0].startswith("Class_"):
        y = np.array([int(label.replace("Class_", "")) for label in y])

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        random_state=seed + 1,
        stratify=y_trainval,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    return train_ds, val_ds, test_ds


def load_telcom_csv(
    path_csv="data/raw/telcom.csv", test_ratio=0.2, val_ratio=0.1, seed=42
):
    df = pd.read_csv(path_csv)

    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df["Churn"] = df["Churn"].astype(int)

    y = df["Churn"].values
    df.drop(columns=["Churn"], inplace=True)

    cat_cols = [col for col in df.columns if df[col].dtype == object]
    num_cols = [col for col in df.columns if df[col].dtype != object]

    if len(cat_cols) > 0:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = ohe.fit_transform(df[cat_cols].fillna("MissingValue"))
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names)
        X_num_df = df[num_cols].copy().fillna(0)
        X_full = pd.concat(
            [X_num_df.reset_index(drop=True), X_cat_df.reset_index(drop=True)], axis=1
        )
    else:
        X_full = df[num_cols].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        random_state=seed + 1,
        stratify=y_trainval,
    )

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    return train_ds, val_ds, test_ds


def load_mnist_csv(
    path_csv="data/raw/mnist_784.csv", test_ratio=0.2, val_ratio=0.1, seed=42
):
    df = pd.read_csv(path_csv)

    target_col = "class" if "class" in df.columns else "label"

    y = df[target_col].values
    df.drop(columns=[target_col], inplace=True)

    if y.dtype == object:
        y = np.array([int(label) for label in y])
    else:
        y = y.astype(int)

    X = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        random_state=seed + 1,
        stratify=y_trainval,
    )

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    return train_ds, val_ds, test_ds


def load_dataset(dataset_name, test_ratio=0.2, val_ratio=0.1, seed=42):
    if dataset_name == "california":
        return load_california_housing(
            test_ratio=test_ratio, val_ratio=val_ratio, seed=seed
        )
    elif dataset_name == "otto":
        return load_otto_csv(test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)
    elif dataset_name == "telcom":
        return load_telcom_csv(test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)
    elif dataset_name == "mnist":
        return load_mnist_csv(test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)
    elif dataset_name == "wine":
        return load_wine_quality(test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)
    elif dataset_name == "covertype":
        return load_covertype_dataset(
            test_ratio=test_ratio, val_ratio=val_ratio, seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
