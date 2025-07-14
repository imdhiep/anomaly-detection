import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from patch_extraction import extract_patches_with_coordinatess


class FeatureEmbedding(nn.Module):
    """Convert sampled patches from CNN feature maps into fixed-length vectors."""
    
    def __init__(self, 
                 input_channels: int,
                 patch_size: Tuple[int, int, int],
                 embedding_method: str = 'flatten',
                 pooling_type: str = 'avg',
                 mlp_hidden_dims: Optional[List[int]] = None,
                 output_dim: Optional[int] = None):
        """
        Initialize feature embedding module.
        
        Args:
            input_channels: Number of input channels from feature maps
            patch_size: Size of patches (D, H, W)
            embedding_method: 'flatten', 'pool', or 'mlp'
            pooling_type: 'avg', 'max', or 'adaptive' (if embedding_method is 'pool')
            mlp_hidden_dims: Hidden dimensions for MLP projection
            output_dim: Output dimension for MLP projection
        """
        super(FeatureEmbedding, self).__init__()
        
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embedding_method = embedding_method
        self.pooling_type = pooling_type
        
        # Calculate flattened dimension
        self.flattened_dim = input_channels * patch_size[0] * patch_size[1] * patch_size[2]
        
        if embedding_method == 'flatten':
            self.embed_dim = self.flattened_dim
        elif embedding_method == 'pool':
            if pooling_type == 'adaptive':
                self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.embed_dim = input_channels
            elif pooling_type == 'avg':
                self.pool = nn.AvgPool3d(patch_size)
                self.embed_dim = input_channels
            elif pooling_type == 'max':
                self.pool = nn.MaxPool3d(patch_size)
                self.embed_dim = input_channels
            else:
                raise ValueError(f"Unknown pooling type: {pooling_type}")
        elif embedding_method == 'mlp':
            if mlp_hidden_dims is None or output_dim is None:
                raise ValueError("mlp_hidden_dims and output_dim must be specified for MLP embedding")
            
            layers = []
            prev_dim = self.flattened_dim
            
            for hidden_dim in mlp_hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.mlp = nn.Sequential(*layers)
            self.embed_dim = output_dim
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")
    
    def forward(self, feature_patches: torch.Tensor) -> torch.Tensor:
        """
        Convert feature patches to embeddings.
        
        Args:
            feature_patches: Tensor of shape (N, C, D, H, W) where N is number of patches
        
        Returns:
            Embedded features of shape (N, embed_dim)
        """
        if self.embedding_method == 'flatten':
            # Flatten spatial dimensions
            embeddings = feature_patches.view(feature_patches.size(0), -1)
        
        elif self.embedding_method == 'pool':
            # Apply pooling
            pooled = self.pool(feature_patches)
            embeddings = pooled.view(pooled.size(0), -1)
        
        elif self.embedding_method == 'mlp':
            # Flatten then apply MLP
            flattened = feature_patches.view(feature_patches.size(0), -1)
            embeddings = self.mlp(flattened)
        
        return embeddings


class PatchSampler(nn.Module):
    """Sample patches from CNN feature maps and convert to embeddings."""
    
    def __init__(self,
                 patch_size: Tuple[int, int, int],
                 stride: Optional[Tuple[int, int, int]] = None,
                 max_patches: Optional[int] = None,
                 sampling_strategy: str = 'uniform'):
        """
        Initialize patch sampler.
        
        Args:
            patch_size: Size of patches to extract
            stride: Stride for patch extraction
            max_patches: Maximum number of patches to sample
            sampling_strategy: 'uniform', 'random', or 'grid'
        """
        super(PatchSampler, self).__init__()
        
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches
        self.sampling_strategy = sampling_strategy
    
    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample patches from feature map.
        
        Args:
            feature_map: Feature map tensor of shape (C, D, H, W)
        
        Returns:
            Tuple of (patches, coordinates)
            patches: Tensor of shape (N, C, patch_d, patch_h, patch_w)
            coordinates: Tensor of shape (N, 3)
        """
        if feature_map.dim() != 4:
            raise ValueError(f"Expected 4D feature map, got {feature_map.dim()}D")
        
        # Extract all possible patches with coordinates
        patches_flat, coordinates = extract_patches_with_coordinates(
            feature_map.unsqueeze(0),  # Add batch dimension
            self.patch_size,
            stride=self.stride,
            return_coordinates=True
        )
        
        # Reshape patches back to 5D
        C = feature_map.size(0)
        patch_d, patch_h, patch_w = self.patch_size
        patches = patches_flat.view(-1, C, patch_d, patch_h, patch_w)
        
        # Apply sampling strategy
        if self.max_patches is not None and patches.size(0) > self.max_patches:
            if self.sampling_strategy == 'uniform':
                # Uniform sampling
                indices = torch.linspace(0, patches.size(0) - 1, self.max_patches, dtype=torch.long)
            elif self.sampling_strategy == 'random':
                # Random sampling
                indices = torch.randperm(patches.size(0))[:self.max_patches]
            elif self.sampling_strategy == 'grid':
                # Grid-based sampling
                step = patches.size(0) // self.max_patches
                indices = torch.arange(0, patches.size(0), step)[:self.max_patches]
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            
            patches = patches[indices]
            coordinates = coordinates[indices]
        
        return patches, coordinates


class MultiScaleFeatureEmbedding(nn.Module):
    """Extract and embed features at multiple scales."""
    
    def __init__(self,
                 feature_extractor,
                 patch_sizes: List[Tuple[int, int, int]],
                 embedding_configs: List[Dict],
                 layer_names: List[str],
                 fusion_method: str = 'concat'):
        """
        Initialize multi-scale feature embedding.
        
        Args:
            feature_extractor: Pre-trained feature extractor
            patch_sizes: List of patch sizes for each scale
            embedding_configs: List of embedding configurations
            layer_names: List of layer names to extract features from
            fusion_method: 'concat', 'sum', or 'attention'
        """
        super(MultiScaleFeatureEmbedding, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.patch_sizes = patch_sizes
        self.layer_names = layer_names
        self.fusion_method = fusion_method
        
        # Create patch samplers and embeddings for each scale
        self.patch_samplers = nn.ModuleList()
        self.embeddings = nn.ModuleList()
        
        for patch_size, config in zip(patch_sizes, embedding_configs):
            sampler = PatchSampler(patch_size, **config.get('sampler_args', {}))
            embedding = FeatureEmbedding(patch_size=patch_size, **config.get('embedding_args', {}))
            
            self.patch_samplers.append(sampler)
            self.embeddings.append(embedding)
        
        # Calculate total embedding dimension
        self.total_embed_dim = sum(emb.embed_dim for emb in self.embeddings)
        
        if fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=self.embeddings[0].embed_dim,
                num_heads=8,
                batch_first=True
            )
    
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features from volume.
        
        Args:
            volume: Input volume tensor
        
        Returns:
            Multi-scale feature embeddings
        """
        # Extract feature maps
        feature_maps = self.feature_extractor.extract_features(volume, self.layer_names)
        
        all_embeddings = []
        
        for layer_name in self.layer_names:
            feature_map = feature_maps[layer_name]
            
            layer_embeddings = []
            for sampler, embedding in zip(self.patch_samplers, self.embeddings):
                # Sample patches
                patches, _ = sampler(feature_map.squeeze(0))  # Remove batch dim
                
                # Embed patches
                embedded = embedding(patches)
                layer_embeddings.append(embedded)
            
            # Fuse embeddings from different scales
            if self.fusion_method == 'concat':
                fused = torch.cat(layer_embeddings, dim=1)
            elif self.fusion_method == 'sum':
                # Ensure all embeddings have the same dimension
                min_dim = min(emb.size(1) for emb in layer_embeddings)
                trimmed = [emb[:, :min_dim] for emb in layer_embeddings]
                fused = torch.stack(trimmed).sum(dim=0)
            elif self.fusion_method == 'attention':
                # Use attention to fuse embeddings
                stacked = torch.stack(layer_embeddings, dim=1)  # (N, num_scales, embed_dim)
                fused, _ = self.attention(stacked, stacked, stacked)
                fused = fused.mean(dim=1)  # Average over scales
            
            all_embeddings.append(fused)
        
        # Concatenate embeddings from all layers
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        return final_embeddings
