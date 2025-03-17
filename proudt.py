import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time


class ProuDT:
    """
    Probabilistic Differentiable Decision Tree (ProuDT)

    A user-friendly interface for the differentiable decision tree model.
    """

    def __init__(
        self,
        depth=8,
        learning_rate=0.01,
        max_epochs=10000,
        early_stopping=10,
        random_state=42,
        device=None,
    ):
        """
        Initialize ProuDT model

        Parameters:
        depth: Tree depth
        learning_rate: Learning rate for optimizer
        max_epochs: Maximum training epochs
        early_stopping: Number of epochs with no improvement after which training stops
        random_state: Random seed
        device: 'cuda' or 'cpu' (None for auto-detection)
        """
        self.depth = depth
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Set random seed if provided
        if random_state is not None:
            self._set_seed(random_state)

        self.model = None
        self.scaler = StandardScaler()
        self.num_classes = None

    def _set_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, X_train, y_train, X_val, y_val, verbose=1):
        """
        Train the model on the provided data

        Parameters:
        X_train, y_train: Training data (expects normalized features)
        X_val, y_val: Validation data (expects normalized features)
        verbose: 0 for silent, 1 for progress updates

        Returns:
        self: Trained model
        """
        # Set random seed for reproducibility
        if self.random_state is not None:
            self._set_seed(self.random_state)

        # Convert inputs to numpy if needed
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)

        # Determine number of classes
        self.num_classes = len(np.unique(np.concatenate([y_train, y_val])))

        # Calculate feature importance and get ranking
        mi = mutual_info_classif(X_train, y_train, random_state=self.random_state)
        self.ranked_indices = np.argsort(mi)[::-1].copy()

        # Convert to PyTorch tensors
        device = torch.device(self.device)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

        # Initialize model
        self.model = DifferentiableDecisionTree(
            self.depth, self.num_classes, self.ranked_indices
        ).to(device)

        # Initialize loss function and optimizer
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop with early stopping
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model = None

        for epoch in range(self.max_epochs):
            # Train
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Validate
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

                # Early stopping logic
                if val_loss < best_val_loss - 1e-4:  # Improvement threshold
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model = self.model.state_dict().copy()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.early_stopping:
                    if verbose:
                        print(f"Early stopping at epoch {epoch-self.early_stopping}")
                    break

            # if verbose and (epoch + 1) % 100 == 0:
            #     print(
            #         f"Epoch {epoch+1}/{self.max_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}"
            #     )

        # Load best model
        if best_model:
            self.model.load_state_dict(best_model)

        if verbose:
            # Calculate final metrics
            train_acc = self.score(X_train, y_train)
            val_acc = self.score(X_val, y_val)
            # print(
            #     f"Training complete. Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}"
            # )
            print(f"Training complete. Train accuracy: {train_acc:.4f}")

        return self

    def predict(self, X):
        """
        Make predictions on new data

        Parameters:
        X: Features to predict

        Returns:
        y_pred: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X = np.asarray(X)

        device = torch.device(self.device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X

        Parameters:
        X: Features to predict (expects normalized features)

        Returns:
        probas: Class probabilities for each sample
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X = np.asarray(X)

        device = torch.device(self.device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)

        return probas.cpu().numpy()

    def score(self, X, y):
        """
        Return accuracy score on the given data

        Parameters:
        X: Features
        y: True labels

        Returns:
        score: Accuracy score
        """
        return accuracy_score(y, self.predict(X))

    def evaluate(self, X, y):
        """
        Evaluate model on test data with multiple metrics

        Parameters:
        X: Features
        y: True labels

        Returns:
        metrics: Dictionary containing accuracy, F1-score and inference time
        """
        start_time = time.time()
        y_pred = self.predict(X)
        inference_time = time.time() - start_time

        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")

        return {"accuracy": accuracy, "f1_score": f1, "inference_time": inference_time}


# Required model components (include these in the same file)


class DifferentiableDecisionNode(torch.nn.Module):
    def __init__(self):
        super(DifferentiableDecisionNode, self).__init__()
        # Parameter for the decision threshold
        self.decision = torch.nn.Parameter(torch.randn(1))
        # self.weight=nn.Parameter(torch,randn(1))

    def forward(self, x):
        return torch.sigmoid(self.decision - x)


class DifferentiableDecisionTree(torch.nn.Module):

    def __init__(self, depth, num_classes, ranked_features_indice):
        super(DifferentiableDecisionTree, self).__init__()

        self.depth = depth
        self.num_classes = num_classes
        self.ranked_features_indice = ranked_features_indice
        self.nodes = torch.nn.ModuleList(
            [DifferentiableDecisionNode() for _ in range(2**depth - 1)]
        )
        # Adjusting leaf values to accommodate class scores
        self.leaf_values = torch.nn.Parameter(torch.randn(2**depth, num_classes))

    def forward(self, x):  # fast version
        batch_size, num_features = x.shape
        path_probabilities = torch.ones(batch_size, 2**self.depth, device=x.device)
        node_index = 0
        x = x[:, self.ranked_features_indice]

        for level in range(self.depth):
            level_start = 2**level - 1
            parent_probabilities = path_probabilities.clone()

            indices = torch.arange(2**level, device=x.device)
            node_indices = level_start + indices

            decisions = torch.stack(
                [
                    self.nodes[idx](x[:, idx % num_features]).squeeze()
                    for idx in node_indices
                ],
                dim=1,
            )

            left_children = indices * 2
            right_children = left_children + 1

            path_probabilities[:, left_children] = (
                parent_probabilities[:, indices] * decisions
            )
            path_probabilities[:, right_children] = parent_probabilities[:, indices] * (
                1 - decisions
            )

        output = torch.matmul(path_probabilities, self.leaf_values)
        return output


class FocalLoss(torch.nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate log softmax and get probability for each class
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)

        # Get log probability of correct class for each sample
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Convert to probabilities
        probs = target_log_probs.exp()

        # Focal Loss calculation: FL = -α * (1 - p_t)^γ * log(p_t)
        focal_loss = -self.alpha * (1 - probs) ** self.gamma * target_log_probs

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
