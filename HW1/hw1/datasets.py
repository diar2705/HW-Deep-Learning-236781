import torch
from torch import Tensor
from typing import Tuple, Iterator
from contextlib import contextmanager
from torch.utils.data import Dataset, IterableDataset


def random_labelled_image(
    shape: Tuple[int, ...],
    num_classes: int,
    low=0,
    high=255,
    dtype=torch.int,
) -> Tuple[Tensor, int]:
    """
    Generates a random image and a random class label for it.
    :param shape: The shape of the generated image e.g. (C, H, W).
    :param num_classes: Number of classes. The label should be in [0, num_classes-1].
    :param low: Minimal value in the image (inclusive).
    :param high: Maximal value in the image (exclusive).
    :param dtype: Data type of the returned image tensor.
    :return: A tuple containing the generated image tensor and it's label.
    """

    image = torch.randint(low=low, high=high, size=shape, dtype=dtype)

    label = torch.randint(high=num_classes, size=(1,)).item()

    return image, label


@contextmanager
def torch_temporary_seed(seed: int):
    """
    A context manager which temporarily sets torch's random seed, then sets the random
    number generator state back to its previous state.
    :param seed: The temporary seed to set.
    """
    state = torch.random.get_rng_state()
    try:
        torch.random.manual_seed(seed)
        yield

    finally:
        torch.random.set_rng_state(state)


class RandomImageDataset(Dataset):
    """
    A dataset representing a set of noise images of specified dimensions.
    """

    def __init__(self, num_samples: int, num_classes: int, C: int, W: int, H: int):
        """
        :param num_samples: Number of samples (labeled images in the dataset)
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        :param W: Image width
        :param H: Image height
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_dim = (C, W, H)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Returns a labeled sample.
        :param index: Sample index.
        :return: A tuple (sample, label) containing the image and its class label.
        Raises a ValueError if index is out of range.
        """

        if  not (0 <= index < self.num_samples):
            raise ValueError()
        with torch_temporary_seed(index):
            return random_labelled_image(
                shape=self.image_dim, num_classes=self.num_classes
            )

    def __len__(self):
        """
        :return: Number of samples in this dataset.
        """
        return self.num_samples


class ImageStreamDataset(IterableDataset):
    """
    A dataset representing an infinite stream of noise images of specified dimensions.
    """

    def __init__(self, num_classes: int, C: int, W: int, H: int):
        """
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        :param W: Image width
        :param H: Image height
        """
        super().__init__()
        self.num_classes = num_classes
        self.image_dim = (C, W, H)

    def __iter__(self) -> Iterator[Tuple[Tensor, int]]:
        """
        :return: An iterator providing an infinite stream of random labelled images.
        """

        while True:
            yield random_labelled_image(
                shape=self.image_dim, num_classes=self.num_classes
            )


class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset, returning a subset from it.
    """

    def __init__(self, source_dataset: Dataset, subset_len: int, offset=0):
        """
        Create a SubsetDataset from another dataset.
        :param source_dataset: The dataset to take samples from.
        :param subset_len: The total number of sample in the subset.
        :param offset: The offset index to start taking samples from.
        """
        if offset + subset_len > len(source_dataset):
            raise ValueError("Not enough samples in source dataset")

        self.source_dataset = source_dataset
        self.subset_len = subset_len
        self.offset = offset

    def __getitem__(self, index):
        if not (0 <= index < self.subset_len):
            raise IndexError()
        return self.source_dataset.__getitem__(index + self.offset)

    def __len__(self):
        return self.subset_len
