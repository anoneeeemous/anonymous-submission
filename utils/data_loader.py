import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from utils.config import dataset_list_small, dataset_list_large


def load_dataset(name, seed):
    """
    Load dataset and perform necessary preprocessing

    Parameters:
    name: Dataset name
    seed: Random seed

    Returns:
    X_train, y_train, X_valid, y_valid, X_test, y_test, ranked_indices
    """
    if name in dataset_list_small:
        # Small dataset processing
        X, y = load_dataset_DGT(name)
        y = LabelEncoder().fit_transform(y)

        (X_train, y_train, X_valid, y_valid, X_test, y_test) = split_train_test_valid(
            X,
            y,
            valid_frac=0.25,
            test_frac=0.25,
            seed=seed,
        )

    else:
        if name == "connect-4":
            X, y = load_dataset_plainFile("connect-4")  # 0.64:0.16:0.2
            y = LabelEncoder().fit_transform(y)

            (X_train, y_train, X_valid, y_valid, X_test, y_test) = (
                split_train_test_valid_ratio(
                    X, y, valid_ratio=0.2, test_ratio=0.2, seed=seed
                )
            )
        if name == "segment":
            X, y = load_dataset_plainFile("segment")  # 0.64:0.16:0.2
            y = LabelEncoder().fit_transform(y)

            (X_train, y_train, X_valid, y_valid, X_test, y_test) = (
                split_train_test_valid_ratio(
                    X, y, valid_ratio=0.2, test_ratio=0.2, seed=seed
                )
            )
        if name == "letter":
            X_train, y_train = load_dataset_plainFile("letter_train")
            X_test, y_test = load_dataset_plainFile("letter_test")
            X_valid, y_valid = load_dataset_plainFile("letter_val")

            print(
                f"X_train shape: {X_train.shape},X_valid shape: {X_valid.shape},X_test shape:{X_test.shape}"
            )
            encoder = LabelEncoder()
            # Fit on y_train
            y_train = encoder.fit_transform(y_train)
            # Transform y_valid and y_test
            y_valid = encoder.transform(y_valid)
            y_test = encoder.transform(y_test)
        if name == "satImage":
            X_train, y_train = load_dataset_plainFile("satImage_train")
            X_test, y_test = load_dataset_plainFile("satImage_test")
            X_valid, y_valid = load_dataset_plainFile("satImage_val")

            print(
                f"X_train shape: {X_train.shape},X_valid shape: {X_valid.shape},X_test shape:{X_test.shape}"
            )
            encoder = LabelEncoder()
            # Fit on y_train
            y_train = encoder.fit_transform(y_train)
            # Transform y_valid and y_test
            y_valid = encoder.transform(y_valid)
            y_test = encoder.transform(y_test)

        if name == "pendigits":
            X_train_valid, y_train_valid = load_dataset_plainFile("pendigit_train")
            X_test, y_test = load_dataset_plainFile("pendigit_test")
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_valid,
                y_train_valid,
                test_size=0.2,
                stratify=y_train_valid,
                random_state=seed,
            )

            print(
                f"X_train shape: {X_train.shape},X_valid shape: {X_valid.shape},X_test shape:{X_test.shape}"
            )
            encoder = LabelEncoder()
            # Fit on y_train
            y_train = encoder.fit_transform(y_train)
            # Transform y_valid and y_test
            y_valid = encoder.transform(y_valid)
            y_test = encoder.transform(y_test)

        if name == "mnist":
            # X_train_valid, y_train_valid, X_test, y_test = load_dataset_MNIST()
            X_train_valid, y_train_valid, X_test, y_test = load_dataset_MNIST()
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_valid,
                y_train_valid,
                test_size=0.2,
                stratify=y_train_valid,
                random_state=seed,
            )
            encoder = LabelEncoder()
            # Fit on y_train
            y_train = encoder.fit_transform(y_train)
            # Transform y_valid and y_test
            y_valid = encoder.transform(y_valid)
            y_test = encoder.transform(y_test)
            print(
                f"X_train shape: {X_train.shape},X_valid shape: {X_valid.shape},X_test shape:{X_test.shape}"
            )
        if name == "protein":
            X_train, y_train = load_LIBSVM_dataset(
                "datasets/protein/protein_clean.tr.bz2"
            )
            X_valid, y_valid = load_LIBSVM_dataset(
                "datasets/protein/protein_clean.val.bz2"
            )
            X_test, y_test = load_LIBSVM_dataset("datasets/protein/protein_clean.t.bz2")
            print(
                f"X_train shape: {X_train.shape},X_valid shape: {X_valid.shape},X_test shape:{X_test.shape}"
            )

        if name == "senseIT":
            X_train_valid, y_train_valid = load_LIBSVM_dataset(
                "datasets/senseIT/combined_scale.bz2"
            )

            X_test, y_test = load_LIBSVM_dataset(
                "datasets/senseIT/combined_scale.t.bz2"
            )

            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_valid,
                y_train_valid,
                test_size=0.2,
                stratify=y_train_valid,
                random_state=seed,
            )
            encoder = LabelEncoder()
            # Fit on y_train
            y_train = encoder.fit_transform(y_train)
            # Transform y_valid and y_test
            y_valid = encoder.transform(y_valid)
            y_test = encoder.transform(y_test)
            print(
                f"X_train shape: {X_train.shape},X_valid shape: {X_valid.shape},X_test shape:{X_test.shape}"
            )

    # # Ensure labels are encoded
    # label_encoder = LabelEncoder()
    # y_train = label_encoder.fit_transform(y_train)
    # y_valid = label_encoder.transform(y_valid)
    # y_test = label_encoder.transform(y_test)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # Calculate feature importance
    mi = mutual_info_classif(X_train, y_train, random_state=seed)
    ranked_indices = np.argsort(mi)[::-1].copy()

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return (
        X_train_tensor,
        y_train_tensor,
        X_valid_tensor,
        y_valid_tensor,
        X_test_tensor,
        y_test_tensor,
        ranked_indices,
    )


def load_dataset_DGT(name):
    try:
        if name == "acute-inflammations-1":
            filepath = "datasets/acute-inflammations-1.data"
        elif name == "acute-inflammations-2":
            filepath = "datasets/acute-inflammations-2.data"
        elif name == "avila":
            filepath = "datasets/avila-tr.txt"
        elif name == "balance-scale":
            filepath = "datasets/balance-scale.data"
        elif name == "blood-transfusion":
            filepath = "datasets/blood-transfusion.data"
        elif name == "breast-cancer-wisconsin":
            filepath = "datasets/breast-cancer-wisconsin.data"
        elif name == "car":
            filepath = "datasets/car.data"
        elif name == "climate_crashes":
            filepath = "datasets/climate-crashes.data"
        elif name == "banknote_authentication":
            filepath = "datasets/data_banknote_authentication.txt"
        elif name == "drybean":
            filepath = "datasets/drybean_data.txt"
        # elif name == "house-votes-84":
        #     filepath = "datasets_DGT/house-votes-84.data"
        elif name == "optical-recognition":
            filepath = "datasets/optdigits.tra"
        elif name == "sonar":
            filepath = "datasets/sonar.all-data"
        elif name == "wine-red":
            filepath = "datasets/winequality-red.csv"
        elif name == "wine-white":
            filepath = "datasets/winequality-white.csv"
        else:
            print("no dataset filepath for this name")

        import os

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file at {filepath} does not exist.")

        Xy = np.loadtxt(open(filepath, "r"), delimiter=",")
    except FileNotFoundError:
        print("File not found")

    X = Xy[:, :-1]
    y = Xy[:, -1]
    print("X shape: ", X.shape)
    # y = [[i for i, (c, _) in enumerate(config["classes"]) if c == y_i][0] for y_i in y]
    return np.array(X), np.array(y)


def load_dataset_plainFile(name):
    from sklearn.datasets import load_svmlight_file

    # Path to your dataset file
    file_path = f"datasets/{name}"

    # Load the dataset
    X, y = load_svmlight_file(file_path)

    # Convert the sparse matrix to a dense array for better visualization
    X_dense = X.toarray()
    # Print the shape of the feature matrix and labels
    print(f"Shape of X after initial loading: {X_dense.shape}")
    print(f"Shape of y after initial loading: {y.shape}")
    return X_dense, y


def load_dataset_MNIST():

    train_data = np.loadtxt("datasets/mnist/mnist_train.txt", delimiter=",")
    test_data = np.loadtxt("datasets/mnist/mnist_test.txt", delimiter=",")
    # split features and labels
    y_train = train_data[:, 0]  # 1st column is the label
    X_train = train_data[:, 1:]

    y_test = test_data[:, 0]
    X_test = test_data[:, 1:]

    # print shapes to verify
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test


def load_LIBSVM_dataset(file_path):
    import bz2
    import numpy as np
    from sklearn.datasets import load_svmlight_file

    """
  
    Returns:
        X_dense (numpy.ndarray): Features as a dense NumPy array.
        y (numpy.ndarray): Labels as a NumPy array.
    """

    X, y = load_svmlight_file(file_path)

    # convert sparse matrix to dense format
    X_dense = X.toarray()
    return X_dense, y


def split_train_test_valid(
    X_data, y_data, valid_frac=0.25, test_frac=0.25, seed=42, verbosity=0
):
    data_size = X_data.shape[0]
    test_size = int(data_size * test_frac)
    valid_size = int(data_size * valid_frac)
    print("test_ratio:", test_frac, " valid_ratio is: ", valid_frac)

    X_train_with_valid, X_test, y_train_with_valid, y_test = train_test_split(
        X_data, y_data, test_size=test_size, stratify=y_data, random_state=seed
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_with_valid,
        y_train_with_valid,
        test_size=valid_size,
        stratify=y_train_with_valid,
        random_state=seed,
    )

    if verbosity > 0:
        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)
        print(X_test.shape, y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def split_train_test_valid_ratio(
    X_data, y_data, valid_ratio=0.2, test_ratio=0.2, seed=42
):

    X_train_with_valid, X_test, y_train_with_valid, y_test = train_test_split(
        X_data, y_data, test_size=test_ratio, stratify=y_data, random_state=seed
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_with_valid,
        y_train_with_valid,
        test_size=valid_ratio,
        stratify=y_train_with_valid,
        random_state=seed,
    )
    return X_train, y_train, X_valid, y_valid, X_test, y_test
