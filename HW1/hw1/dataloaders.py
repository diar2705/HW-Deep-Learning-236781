import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler, DataLoader
import torch.utils.data.dataloader
import torch.utils.data.sampler as torch_sampler

from torch.utils.data.sampler import SubsetRandomSampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        low = 0
        high = self.__len__() - 1
        flag = 0
        while low <= high:
            if flag == 0:
                flag = 1
                low += 1
                yield low - 1
            elif flag == 1:
                flag = 0
                high -= 1
                yield high + 1

    def __len__(self):
        return len(self.data_source)


def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    validation_size = int(validation_ratio * len(dataset))

    train_sampler = SubsetRandomSampler(
        [i for i in range(validation_size, len(dataset))]
    )
    validation_sampler = SubsetRandomSampler([i for i in range(validation_size)])

    dl_train = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        sampler=train_sampler,
        batch_size=batch_size,
    )
    dl_valid = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        sampler=validation_sampler,
        batch_size=batch_size,
    )

    return dl_train, dl_valid
