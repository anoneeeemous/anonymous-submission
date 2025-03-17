import torch
import torch.nn as nn
import math


class DifferentiableDecisionNode(nn.Module):
    def __init__(self):
        super(DifferentiableDecisionNode, self).__init__()
        # Parameter for the decision threshold
        self.decision = nn.Parameter(torch.randn(1))
        # self.weight=nn.Parameter(torch,randn(1))

    def forward(self, x):
        return torch.sigmoid(self.decision - x)


class DifferentiableDecisionTree(nn.Module):

    def __init__(self, depth, num_classes, ranked_features_indice):
        super(DifferentiableDecisionTree, self).__init__()

        self.depth = depth
        self.num_classes = num_classes
        self.ranked_features_indice = ranked_features_indice
        self.nodes = nn.ModuleList(
            [DifferentiableDecisionNode() for _ in range(2**depth - 1)]
        )
        # Adjusting leaf values to accommodate class scores
        self.leaf_values = nn.Parameter(torch.randn(2**depth, num_classes))

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
