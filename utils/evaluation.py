import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, X, y, num_classes):
    """
    Evaluate model performance

    Parameters:
    model: Model to evaluate
    X: Feature data
    y: Label data
    num_classes: Number of classes

    Returns:
    Accuracy, F1 score and computation time
    """
    with torch.no_grad():
        # Calculate accuracy and time it
        start_time = time.time()
        predictions = model(X)
        probabilities = torch.softmax(predictions, dim=1)
        predicted_classes = probabilities.argmax(dim=1)

        # Convert to CPU for sklearn compatibility
        y_true = y.cpu().numpy()
        y_pred = predicted_classes.cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        accuracy_time = time.time() - start_time

        f1_macro = f1_score(y_true, y_pred, average="macro")

        return accuracy, f1_macro, accuracy_time


def set_seed(seed):
    """
    Set random seed for reproducibility

    Parameters:
    seed: Random seed
    """
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
