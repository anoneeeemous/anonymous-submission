# datasets from UCI
dataset_list_small = [
    "breast-cancer-wisconsin",
    "car",
    "banknote_authentication",
    "balance-scale",
    "acute-inflammations-1",
    "acute-inflammations-2",
    "blood-transfusion",
    "climate_crashes",
    "sonar",
    "optical-recognition",
    "drybean",
    "avila",
    "wine-red",
    "wine-white",
]

# datasets from LIBSVM
dataset_list_large = [
    "connect-4",
    "mnist",
    "segment",
    "letter",
    "satImage",
    "pendigits",
    "protein",
    "senseIT",
]

# All datasets list
all_datasets = dataset_list_small + dataset_list_large

# Default configuration parameters
default_config = {
    # Model parameters
    "small_depth": 8,  # Default
    "large_depth": 11,  # Default
    "small_trials": 100,  # Default
    "large_trials": 10,  # Default
    # Training parameters
    "max_epochs": 10000,  # Maximum training epochs
    "patience": 10,  # Early stopping patience
    "threshold": 0.0001,  # Early stopping threshold
    "learning_rate": 0.01,  # Learning rate
    "focal_alpha": 1,  # Focal loss alpha parameter
    "focal_gamma": 2,  # Focal loss gamma parameter
    "valid_ratio": 0.25,
    "test_ratio": 0.25,
    "large_valid_ratio": 0.2,
    "large_test_ratio": 0.2,
    # Result storage
    "result_dir": "results",  # Results directory
    "batchSize": 512,
}
