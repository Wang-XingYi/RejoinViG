import torch
import torch.nn as nn
import torch.nn.functional as F


# 针对5分类任务的 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, size_average=True):
        """
        Args:
            alpha: 平衡因子，可以是一个标量或者一个长度为 num_classes 的列表。
            gamma: 调节因子，控制对难分类样本的关注程度。
            size_average: 是否对 batch 内的 loss 求平均。
        """
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        """
        Args:
            pred: 模型的预测值，形状为 (BatchSize, num_classes)。
            target: 真实标签，形状为 (BatchSize,)。
        Returns:
            Focal Loss 值。
        """
        # 对 pred 计算 softmax，获得每个类别的概率
        pred_softmax = F.softmax(pred, dim=1)
        # 确保 target 的数据类型为 long
        target = target.long()

        # 将 target 转为 one-hot 编码，形状为 (BatchSize, num_classes)
        target_one_hot = torch.zeros_like(pred).scatter_(1, target.view(-1, 1), 1)

        # 选出每个样本的预测概率
        probs = (pred_softmax * target_one_hot).sum(dim=1)  # (BatchSize,)

        # 防止数值问题，限制 probs 的范围
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算 log(prob)
        log_p = probs.log()

        # 如果 alpha 是一个标量，扩展为与 num_classes 维度一致
        if isinstance(self.alpha, (float, int)):
            alpha = torch.tensor([1 - self.alpha] * pred.size(1)).cuda()
            alpha[-1] = self.alpha
        else:
            alpha = self.alpha


        # 利用 target_one_hot 选择对应类别的 alpha
        alpha_t = (alpha * target_one_hot).sum(dim=1)  # (BatchSize,)

        # 根据 Focal Loss 公式计算 loss
        focal_loss = -alpha_t * (torch.pow((1 - probs), self.gamma)) * log_p

        # 根据 size_average 决定返回 mean 或 sum
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


