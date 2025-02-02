import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        channels = [in_channels, 64, 128, 256, 512, out_channels]
        for i in range(len(channels) - 1):
            modules.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            modules.append(nn.BatchNorm2d(channels[i + 1]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout2d(0.1))

        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        channels = [in_channels, 512, 256, 128, 64, out_channels]
        for i in range(len(channels) - 1):
            modules.append(
                nn.ConvTranspose2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if i < len(channels) - 2:
                modules.append(nn.BatchNorm2d(num_features=channels[i + 1]))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout2d(0.1))
            else:
                modules.append(nn.Tanh())

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        return self.cnn(h)


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        self.mu = nn.Linear(n_features, z_dim)
        self.log_var = nn.Linear(n_features, z_dim)
        self.trans = nn.Linear(z_dim, n_features)

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        h = self.features_encoder(x).view(x.size(0), -1)
        mu = self.mu(h)
        log_sigma2 = self.log_var(h)
        std = torch.exp(0.5 * log_sigma2)
        z = mu + std * torch.randn_like(mu)

        return z, mu, log_sigma2

    def decode(self, z):
        x_rec = self.features_decoder(self.trans(z).reshape(-1, *self.features_shape))
        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            for _ in range(n):
                z = torch.randn((1, self.z_dim), device=device)
                samples.append(self.decode(z).squeeze(0))

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None

    norm = torch.norm((x - xr).view(x.size(0), -1), dim=1).pow(2)
    data_loss = torch.mean(norm) / (x[0].numel() * x_sigma2)
    kldiv_loss = torch.mean(
        z_log_sigma2.exp().sum(1)
        + torch.norm(z_mu, dim=1).pow(2)
        - z_log_sigma2.sum(1)
        - z_mu.size(1)
    )

    loss = data_loss + kldiv_loss

    return loss, data_loss, kldiv_loss
