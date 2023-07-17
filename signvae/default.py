import torch
import math
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Basic VAE with convolutional encoder and deconvolutional decoder.
    """
    def __init__(self, z_dim):
        """
        Initializes the layers of the VAE, which should include:
        - one dropout layer
        - a stack of convolutional layers (we recommend starting
          with 3 of them) interleaved with max pooling layers
        - a dense layer to project the output from the final
          convolution down to size self.z_dim
        - a dense layer to project the encoder output onto mu
        - a dense layer to project the encoder output onto sigma
        - a stack of deconvolutional layers (AKA transposed convolutional
          layers; we recommend starting with 4 of them) interleaved with
          2d batch normalization layers.

        Input:
        - z_dim:    size of the codes produced by this encoder
        """
        super(VAE, self).__init__()
        self.z_dim = z_dim

        # TODO Your code goes here.

    def encode(self,x):
        """
        Given a sequence of images, applies a stack of convolutional layers to
        each image, flattens the convolved outputs, and projects each one to
        size self.z_dim.

        Input:
        - x:    torch.Tensor of shape (seq_length, n_channels, img_width, img_height)
                which equals (50, 1, 64, 64) with the default model configuration.

        Returns:
        - A torch.Tensor of size (seq_length, self.z_dim) which defaults to (50, 16)
        """
        return torch.rand(x.shape[0], self.z_dim, requires_grad=True).to(x) # TODO Your code goes here.

    def project(self, x):
        """
        Given an intermediate sequence of encoded images, applies two
        projections to each encoding to produce vectors mu and sigma.

        Input:
        - x:    torch.Tensor of shape (seq_length, self.z_dim) (output
                from self.encode)

        Returns:
        - A tuple of two torch.Tensors, each of shape (seq_length, self.z_dim)
        """
        return (x, x) # TODO Your code goes here.

    def reparametrize(self, mu, sigma):
        """
        Applies the reparametrization trick from
        https://arxiv.org/pdf/1312.6114v10.pdf

        Input:
        - mu:       torch.Tensor of shape (seq_length, self.z_dim) returned
                    by self.project()
        - sigma:    torch.Tensor of shape (seq_length, self.z_dim) returned
                    by self.project()

        Returns:
        - A sequence of codes z of shape (seq_length, self.z_dim) obtained by
          sampling from a normal distribution parameterized by mu and sigma
        """
        return mu + sigma # TODO Your code goes here.

    def decode(self, z):
        """
        Given a sequence of variational codes, applies a stack of deconvolutional
        layers (AKA transposed convolutional layers) to recover a sequence of images.

        Input:
        - z:    torch.Tensor of shape (seq_length, self.z_dim) returned by
                self.reparametrize()

        Returns:
        - A sequence of images of shape (seq_length, n_channels, img_width, img_height)
          which defaults to (50, 1, 64, 64). All outputs should be in the range [0, 1].
        """
        return z.repeat_interleave(4, dim=1).unsqueeze(1).repeat(1,64,1).unsqueeze(1) # TODO Your code goes here.

def kld(mu, log_var):
    """
    Computes KL div loss wrt. a standard normal prior.

    Input:
    - log_var:  log variance of encoder outputs
    - mu:       mean of encoder outputs

    Returns:    D_{KL}(\mathcal{N}(mu, sigma) || \mathcal{N}(0, 1))
                = log(1 / sigma) + (sigma^2 + mu^2)/2 - 1/2
                = -0.5*(1 + log(sigma^2) - sigma^2 - mu^2)
    """
    return (mu + log_var).sum() # TODO Your code goes here.

def vae_loss(gen_images, input_images, mu_sigmas):
    """
    Computes BCE reconstruction loss and KL div. loss for VAE outputs.

    Input:
    - gen_images:   list of decoded image sequences, each of shape (seq_length,
                    n_channels, img_width, img_height) which defaults to
                    (50, 1, 64, 64). In the baseline model, this will contain
                    one sequence decoded from the VAE itself, and another decoded
                    from the top layer of the Transformer.
    - input_images: list of target image sequences, each of shape (seq_length,
                    n_channels, img_width, img_height) which defaults to
                    (50, 1, 64, 64). The nth sequence in gen_images will be
                    evaluated against the nth sequence in input_images to
                    compute the reconstruction loss.
    - mu_sigmas:    list of (mu, sigma) tuples, where each mu and sigma is a
                    sequence of shape (seq_length, VAE.z_dim). In the baseline
                    model, this will contain one tuple from the VAE and another
                    from the Transformer.

    Returns:
    - BCEs: a list containing the total BCE reconstruction loss for each image
            sequence
    - KLDs: a list containing the total KL divergence loss for each mu/sigma pair
    """
    # List to aggregate binary cross-entropy reconstruction losses
    # from all of the image outputs:
    BCEs = []
    # List to aggregate KL divergence losses from each of the mu/sigma
    # projections:
    KLDs = []

    # TODO Your code goes here.

    return BCEs, KLDs
