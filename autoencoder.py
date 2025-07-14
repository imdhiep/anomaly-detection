import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union


class Autoencoder(nn.Module):
    """PyTorch autoencoder for feature vectors."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 latent_dim: int,
                 activation: str = 'relu',
                 dropout_rate: float = 0.1,
                 batch_norm: bool = True):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Dimension of input feature vectors
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            activation: Activation function ('relu', 'leaky_relu', 'tanh', 'elu')
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Get activation function
        self.act_fn = self._get_activation_fn(activation)
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
    
    def _get_activation_fn(self, activation: str):
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU(inplace=True))
    
    def _build_encoder(self):
        """Build encoder network."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.act_fn)
            
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, self.latent_dim))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        """Build decoder network."""
        layers = []
        
        # Reverse the hidden dimensions for decoder
        decoder_dims = [self.latent_dim] + list(reversed(self.hidden_dims))
        
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
            
            layers.append(self.act_fn)
            
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
        
        # Final layer to output space
        layers.append(nn.Linear(decoder_dims[-1], self.input_dim))
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed
    
    def get_reconstruction_error(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute reconstruction error."""
        x_reconstructed = self.forward(x)
        
        if reduction == 'none':
            # Return per-sample reconstruction error
            error = F.mse_loss(x_reconstructed, x, reduction='none').sum(dim=1)
        elif reduction == 'mean':
            error = F.mse_loss(x_reconstructed, x, reduction='mean')
        elif reduction == 'sum':
            error = F.mse_loss(x_reconstructed, x, reduction='sum')
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        return error


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for feature vectors."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 latent_dim: int,
                 activation: str = 'relu',
                 dropout_rate: float = 0.1,
                 batch_norm: bool = True,
                 beta: float = 1.0):
        """
        Initialize variational autoencoder.
        
        Args:
            input_dim: Dimension of input feature vectors
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            activation: Activation function
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            beta: Beta parameter for KL divergence weighting
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Get activation function
        self.act_fn = self._get_activation_fn(activation)
        
        # Build encoder
        self.encoder = self._build_encoder(dropout_rate, batch_norm)
        
        # Latent space layers
        encoder_output_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
        
        # Build decoder
        self.decoder = self._build_decoder(dropout_rate, batch_norm)
    
    def _get_activation_fn(self, activation: str):
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU(inplace=True))
    
    def _build_encoder(self, dropout_rate: float, batch_norm: bool):
        """Build encoder network."""
        if not self.hidden_dims:
            return nn.Identity()
        
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.act_fn)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self, dropout_rate: float, batch_norm: bool):
        """Build decoder network."""
        layers = []
        
        # Reverse the hidden dimensions for decoder
        decoder_dims = [self.latent_dim] + list(reversed(self.hidden_dims))
        
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
            
            layers.append(self.act_fn)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Final layer to output space
        layers.append(nn.Linear(decoder_dims[-1], self.input_dim))
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input to latent space parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
    
    def loss_function(self, x: torch.Tensor, x_reconstructed: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> tuple:
        """Compute VAE loss."""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class DenoisingAutoencoder(Autoencoder):
    """Denoising autoencoder for robust feature learning."""
    
    def __init__(self, noise_factor: float = 0.1, **kwargs):
        """
        Initialize denoising autoencoder.
        
        Args:
            noise_factor: Standard deviation of Gaussian noise to add during training
            **kwargs: Arguments for base Autoencoder
        """
        super(DenoisingAutoencoder, self).__init__(**kwargs)
        self.noise_factor = noise_factor
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input."""
        if self.training and self.noise_factor > 0:
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise injection during training."""
        x_noisy = self.add_noise(x)
        z = self.encode(x_noisy)
        x_reconstructed = self.decode(z)
        return x_reconstructed


class ContractiveAutoencoder(Autoencoder):
    """Contractive autoencoder with regularization."""
    
    def __init__(self, lambda_reg: float = 1e-4, **kwargs):
        """
        Initialize contractive autoencoder.
        
        Args:
            lambda_reg: Regularization strength for contractive penalty
            **kwargs: Arguments for base Autoencoder
        """
        super(ContractiveAutoencoder, self).__init__(**kwargs)
        self.lambda_reg = lambda_reg
    
    def compute_contractive_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute contractive penalty."""
        # Compute Jacobian of encoder
        z.backward(torch.ones_like(z), retain_graph=True)
        jacobian = x.grad
        
        # Frobenius norm of Jacobian
        contractive_loss = torch.sum(jacobian ** 2)
        
        return self.lambda_reg * contractive_loss
