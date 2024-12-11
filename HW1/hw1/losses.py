import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        true_classes_scores = torch.gather(x_scores, 1, y.view(-1, 1))
        M = self.delta + x_scores - true_classes_scores
        M = torch.max(M, torch.zeros(M.size()))
        loss = torch.mean(torch.sum(M, 1) - self.delta)

        self.grad_ctx["M"] = M
        self.grad_ctx["X"] = x
        self.grad_ctx["Y"] = y

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).
        """

        M = self.grad_ctx["M"]
        x = self.grad_ctx["X"]
        y = self.grad_ctx["Y"]
        N = x.shape[0]
        indicators = (M > 0).float()
        indicators[range(N), y] -= indicators.sum(dim=1)
        grad = (torch.transpose(x, 0, 1) @ indicators) / N

        return grad
