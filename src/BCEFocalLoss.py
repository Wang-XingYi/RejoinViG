import torch
import torch.nn as nn
import torch.nn.functional as F


# Focal Loss for 5-class classification task
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, size_average=True):
        """
        Args:
            alpha: Balancing factor
            gamma: Focusing parameter
            size_average: Whether to average the loss over the batch
        """
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions, shape (BatchSize, num_classes)
            target: Ground truth labels, shape (BatchSize,)
        Returns:
            Focal Loss value
        """
        pred_softmax = F.softmax(pred, dim=1)
        target = target.long()

        #  Convert target to one-hot encoding, shape (BatchSize, num_classes)
        target_one_hot = torch.zeros_like(pred).scatter_(1, target.view(-1, 1), 1)

        # Extract predicted probabilities for the true classes
        probs = (pred_softmax * target_one_hot).sum(dim=1)  # (BatchSize,)

        # Clamp probs to avoid numerical instability
        probs = probs.clamp(min=0.0001, max=1.0)

        # Compute log(prob)
        log_p = probs.log()

        # If alpha is a scalar, expand it to match the number of classes
        if isinstance(self.alpha, (float, int)):
            alpha = torch.tensor([1 - self.alpha] * pred.size(1)).cuda()
            alpha[-1] = self.alpha
        else:
            alpha = self.alpha


        # Select alpha corresponding to each target class
        alpha_t = (alpha * target_one_hot).sum(dim=1)  # (BatchSize,)

        # Compute focal loss
        focal_loss = -alpha_t * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


