import torch


class TensorView(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, tensor: torch.Tensor):
        return tensor.view(*self.view_dims)


class InvertColors(object):
    """
    Inverts colors in an image given as a tensor.
    """

    def __call__(self, x: torch.Tensor):
        """
        :param x: A tensor of shape (C,H,W) for values in the range [0, 1],
            representing an image.
        :return: The image with inverted colors.
        """
        return 1.0 - x


class FlipUpDown(object):
    def __call__(self, x: torch.Tensor):
        """
        :param x: A tensor of shape (C,H,W) representing an image.
        :return: The image, flipped around the horizontal axis.
        """
        return x.flip(dims=[1])


class BiasTrick(object):
    """
    A transform that applies the "bias trick": Prepends an element equal to
    1 to each sample in a given tensor.
    """

    def __call__(self, x: torch.Tensor):
        """
        :param x: A pytorch tensor of shape (D,) or (N1,...Nk, D).
        We assume D is the number of features and the N's are extra
        dimensions. E.g. shape (N,D) for N samples of D features;
        shape (D,) or (1, D) for one sample of D features.
        :return: A tensor with D+1 features, where a '1' was prepended to
        each sample's feature dimension.
        """
        assert x.dim() > 0, "Scalars not supported"
        
        return torch.cat((torch.ones((*x.shape[:-1], 1), dtype=x.dtype, device=x.device), x), dim=-1)
