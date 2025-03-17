import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss function for handling class imbalance

    Parameters:
    alpha: Weight factor
    gamma: Focusing parameter
    reduction: Loss reduction method ('mean', 'sum', 'none')
    """

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
