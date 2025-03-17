import os
import sys
import time
import argparse
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Add project root directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import DifferentiableDecisionTree
from utils.data_loader import load_dataset
from utils.evaluation import evaluate_model, set_seed
from utils.loss import FocalLoss
from utils.config import (
    dataset_list_small,
    dataset_list_large,
    all_datasets,
    default_config,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ProuDT Experiments")

    parser.add_argument(
        "--datasetname",
        type=str,
        default=None,
        help="Specify dataset name. If not provided, run all datasets.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Specify tree depth. Default is 8 for small datasets, 11 for large datasets.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Specify number of trials. Default is 100 for small datasets, 10 for large datasets.",
    )

    return parser.parse_args()


def run_experiment(name, depth, trials):
    """
    Run experiment for specified dataset

    Parameters:
    name: Dataset name
    depth: Tree depth
    trials: Number of trials
    """
    print(f"Running experiment for dataset {name}, depth = {depth}, trials = {trials}")

    # Ensure results directory exists
    result_dir = os.path.join(default_config["result_dir"], name)
    os.makedirs(result_dir, exist_ok=True)

    all_trials_results = []

    # Run multiple trials based on trials count
    for seed in range(trials):
        print(f"Running trial {seed + 1}/{trials}, seed = {seed}")

        # Load data
        X_train, y_train, X_valid, y_valid, X_test, y_test, ranked_indices = (
            load_dataset(name, seed)
        )

        # Get number of classes
        num_classes = len(torch.unique(torch.cat([y_train, y_valid, y_test])))
        print(f"Number of classes: {num_classes}")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Move data to device
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_valid, y_valid = X_valid.to(device), y_valid.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        # Set random seed
        set_seed(seed)

        # Initialize model
        model = DifferentiableDecisionTree(depth, num_classes, ranked_indices).to(
            device
        )

        # Define loss function and optimizer
        criterion = FocalLoss(
            alpha=default_config["focal_alpha"], gamma=default_config["focal_gamma"]
        )
        optimizer = optim.Adam(model.parameters(), lr=default_config["learning_rate"])

        # Training start time
        start_time = time.time()

        # Training parameters
        max_epochs = default_config["max_epochs"]
        patience = default_config["patience"]
        threshold = default_config["threshold"]
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_weights = None
        optimal_epoch = 0

        # Training loop
        for epoch in range(1, max_epochs + 1):
            # Training mode
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # Validation
            with torch.no_grad():
                val_output = model(X_valid)
                val_loss = criterion(val_output, y_valid).item()

            # Early stopping logic
            if best_val_loss - val_loss > threshold:
                best_val_loss = val_loss
                best_model_weights = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping! Best epoch: {epoch - patience}")
                optimal_epoch = epoch - patience
                break

        # If best model weights exist, load them
        if best_model_weights:
            model.load_state_dict(best_model_weights)
            optimal_epoch = epoch - epochs_no_improve

        # Calculate training time
        train_time = time.time() - start_time

        # Evaluate model
        train_accuracy, train_f1_macro, train_accuracy_time = evaluate_model(
            model, X_train, y_train, num_classes
        )
        test_accuracy, test_f1_macro, test_accuracy_time = evaluate_model(
            model, X_test, y_test, num_classes
        )

        # Store results
        trial_result = {
            "Trial": seed + 1,
            "Seed": seed,
            "name": name,
            "Depth": depth,
            "Optimal Epoch": optimal_epoch,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
            "Train F1 Macro": train_f1_macro,
            "Test F1 Macro": test_f1_macro,
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Train Time": train_time,
            "Test Accuracy Time": test_accuracy_time,
        }
        all_trials_results.append(trial_result)

        # Print results
        print(
            f"Trial {seed+1}, Seed {seed}, Dataset {name}, Depth {depth}, Optimal epoch {optimal_epoch}"
        )
        print(
            f"Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}"
        )
        print(
            f"Train F1 score: {train_f1_macro:.4f}, Test F1 score: {test_f1_macro:.4f}"
        )
        print(f"Train time: {train_time:.2f}s, Test time: {test_accuracy_time:.4f}s")

        # Save current results
        results_df = pd.DataFrame(all_trials_results)
        results_df.to_csv(
            os.path.join(result_dir, f"{name}_all_trials_results.csv"), index=False
        )

    # Calculate statistics
    calculate_statistics(all_trials_results, name)


def calculate_statistics(all_trials_results, dataset_name):
    """
    Calculate and save statistics for experiment results

    Parameters:
    all_trials_results: All experiment results
    dataset_name: Dataset name
    """
    statistics = []
    unique_depths = set(trial["Depth"] for trial in all_trials_results)

    for depth in unique_depths:
        filtered_trials = [
            trial for trial in all_trials_results if trial["Depth"] == depth
        ]

        test_accuracy_results = [trial["Test Accuracy"] for trial in filtered_trials]
        test_f1_results = [trial["Test F1 Macro"] for trial in filtered_trials]
        training_time = [trial["Train Time"] for trial in filtered_trials]
        test_acc_time = [trial["Test Accuracy Time"] for trial in filtered_trials]

        mean_test_accuracy = np.mean(test_accuracy_results)
        std_test_accuracy = np.std(test_accuracy_results)
        mean_f1_score = np.mean(test_f1_results)
        std_f1_score = np.std(test_f1_results)
        mean_training_time = np.mean(training_time)
        mean_test_acc_time = np.mean(test_acc_time)

        statistics.append(
            {
                "Dataset Name": dataset_name,
                "Depth": depth,
                "Mean Test Accuracy": mean_test_accuracy,
                "Std Test Accuracy": std_test_accuracy,
                "Mean F1 Score": mean_f1_score,
                "Std F1 Score": std_f1_score,
                "Mean Training Time": mean_training_time,
                "Mean Test Acc Time": mean_test_acc_time,
            }
        )

    result_dir = os.path.join(default_config["result_dir"], dataset_name)
    statistics_df = pd.DataFrame(statistics)
    statistics_df.to_csv(
        os.path.join(result_dir, f"statistics_summary_{dataset_name}.csv"), index=False
    )

    print(f"Statistics for dataset {dataset_name} saved")
    for depth in unique_depths:
        filtered_stats = [s for s in statistics if s["Depth"] == depth]
        if filtered_stats:
            stats = filtered_stats[0]
            print(
                f"Depth {depth} mean test accuracy: {stats['Mean Test Accuracy']:.4f} ± {stats['Std Test Accuracy']:.4f} "
            )


def main():
    # Parse command line arguments
    args = parse_args()

    # Determine datasets to run
    if args.datasetname:
        if args.datasetname in all_datasets:
            datasets = [args.datasetname]
        else:
            print(f"Error: Unknown dataset {args.datasetname}")
            return
    else:
        # If no dataset specified, run all datasets
        datasets = all_datasets

    # Run each dataset sequentially
    for name in datasets:
        # Determine depth and number of trials
        if name in dataset_list_small:
            depth = (
                args.depth if args.depth is not None else default_config["small_depth"]
            )
            trials = (
                args.trials
                if args.trials is not None
                else default_config["small_trials"]
            )
        else:
            depth = (
                args.depth if args.depth is not None else default_config["large_depth"]
            )
            trials = (
                args.trials
                if args.trials is not None
                else default_config["large_trials"]
            )

        # Run experiment
        run_experiment(name, depth, trials)


if __name__ == "__main__":
    main()
