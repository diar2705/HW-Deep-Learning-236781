import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        self.weights = torch.randn((self.n_features,self.n_classes)) * weight_std
        
    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        class_scores = x @ self.weights
        _, y_pred = torch.max(class_scores, dim=1)
        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        acc = (y == y_pred).float().mean()
        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0
            for b_x, b_y in dl_train:
                y_pred, x_scores = self.predict(b_x)
                total_correct += self.evaluate_accuracy(b_y, y_pred)
                average_loss += loss_fn.loss(b_x, b_y, x_scores, y_pred)
                self.weights -= learn_rate * loss_fn.grad() + weight_decay * self.weights
            
            train_res.accuracy.append(total_correct / len(dl_train))
            train_res.loss.append(average_loss / len(dl_train))
            total_correct = 0
            average_loss = 0
            
            for b_x, b_y in dl_valid:
                y_pred, x_scores = self.predict(b_x)
                total_correct += self.evaluate_accuracy(b_y, y_pred)
                average_loss += loss_fn.loss(b_x, b_y, x_scores, y_pred)
            
            valid_res.accuracy.append(total_correct / len(dl_valid))
            valid_res.loss.append(average_loss / len(dl_valid))
            
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """
        weights = self.weights[1:] if has_bias else self.weights
        w_images = weights.T.reshape((self.n_classes, *img_shape))

        return w_images


def hyperparams():
    hp = dict(weight_std=0.001, learn_rate=0.01, weight_decay=0.001)

    return hp
