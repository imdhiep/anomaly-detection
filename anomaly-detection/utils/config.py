import torch
import os
import yaml
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import logging


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_channels: int = 4
    feature_layers: List[str] = None
    feature_size: Tuple[int, int] = (64, 64)
    latent_dim: int = 128
    input_size: Tuple[int, int] = (256, 256)
    reduce_channels: bool = True
    reduction_factor: int = 4
    
    def __post_init__(self):
        if self.feature_layers is None:
            self.feature_layers = ["layer1", "layer2", "layer3", "layer4"]


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 700
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_interval: int = 50
    validation_interval: int = 10
    early_stopping_patience: int = 100
    gradient_clip_norm: float = 1.0


@dataclass
class DataConfig:
    """Data configuration"""
    train_normal_dir: str = "data/train/normal"
    val_normal_dir: str = "data/val/normal"
    test_dir: str = "data/test"
    simulate_depth: bool = True
    use_reflection_padding: bool = False
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    anomaly_threshold: float = 0.5
    save_visualizations: bool = True
    visualization_samples: int = 10
    compute_pro_auc: bool = True
    pro_num_thresholds: int = 100


class ExperimentConfig:
    """Complete experiment configuration"""
    
    def __init__(self, 
                 model_config: ModelConfig = None,
                 training_config: TrainingConfig = None,
                 data_config: DataConfig = None,
                 evaluation_config: EvaluationConfig = None,
                 experiment_name: str = "default"):
        
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        self.data = data_config or DataConfig()
        self.evaluation = evaluation_config or EvaluationConfig()
        self.experiment_name = experiment_name
    
    def save_config(self, save_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'evaluation': asdict(self.evaluation),
            'experiment_name': self.experiment_name
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_config(cls, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        data_config = DataConfig(**config_dict['data'])
        evaluation_config = EvaluationConfig(**config_dict['evaluation'])
        experiment_name = config_dict.get('experiment_name', 'default')
        
        return cls(model_config, training_config, data_config, 
                  evaluation_config, experiment_name)


def create_rgb_only_config() -> ExperimentConfig:
    """Create configuration for RGB-only input"""
    model_config = ModelConfig(input_channels=3)
    return ExperimentConfig(
        model_config=model_config,
        experiment_name="rgb_only"
    )


def create_rgbd_config() -> ExperimentConfig:
    """Create configuration for RGB-D input"""
    model_config = ModelConfig(input_channels=4)
    return ExperimentConfig(
        model_config=model_config,
        experiment_name="rgbd_full"
    )


def create_reduced_layers_config(layers: List[str]) -> ExperimentConfig:
    """Create configuration with specific feature layers"""
    model_config = ModelConfig(
        input_channels=4,
        feature_layers=layers
    )
    layer_str = "_".join(layers)
    return ExperimentConfig(
        model_config=model_config,
        experiment_name=f"rgbd_{layer_str}"
    )


def create_ablation_configs() -> Dict[str, ExperimentConfig]:
    """Create configurations for ablation studies"""
    configs = {}
    
    # RGB vs RGB-D
    configs["rgb_only"] = create_rgb_only_config()
    configs["rgbd_full"] = create_rgbd_config()
    
    # Different layer combinations
    configs["rgbd_layer12"] = create_reduced_layers_config(["layer1", "layer2"])
    configs["rgbd_layer23"] = create_reduced_layers_config(["layer2", "layer3"])
    configs["rgbd_layer234"] = create_reduced_layers_config(["layer2", "layer3", "layer4"])
    configs["rgbd_layer34"] = create_reduced_layers_config(["layer3", "layer4"])
    
    # Different latent dimensions
    for latent_dim in [64, 128, 256]:
        model_config = ModelConfig(input_channels=4, latent_dim=latent_dim)
        configs[f"rgbd_latent{latent_dim}"] = ExperimentConfig(
            model_config=model_config,
            experiment_name=f"rgbd_latent{latent_dim}"
        )
    
    return configs


class Logger:
    """Enhanced logging utility"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file logging
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(experiment_name)
    
    def log_config(self, config: ExperimentConfig):
        """Log experiment configuration"""
        self.logger.info(f"Starting experiment: {config.experiment_name}")
        self.logger.info(f"Model config: {asdict(config.model)}")
        self.logger.info(f"Training config: {asdict(config.training)}")
        self.logger.info(f"Data config: {asdict(config.data)}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics"""
        for metric_name, value in metrics.items():
            self.logger.info(f"{prefix}{metric_name}: {value:.4f}")
    
    def log_training_step(self, epoch: int, train_loss: float, 
                         val_loss: float = None):
        """Log training step"""
        if val_loss is not None:
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}")
        else:
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")


def setup_experiment_directory(experiment_name: str, 
                              base_dir: str = "experiments") -> str:
    """Setup experiment directory structure"""
    exp_dir = os.path.join(base_dir, experiment_name)
    
    # Create subdirectories
    subdirs = ["checkpoints", "logs", "visualizations", "configs", "results"]
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir


def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    device_info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) 
                         for i in range(torch.cuda.device_count())],
            "current_device": torch.cuda.current_device(),
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved()
        })
    
    return device_info


def save_experiment_summary(config: ExperimentConfig, 
                           metrics: Dict[str, float],
                           experiment_dir: str):
    """Save experiment summary"""
    summary = {
        "experiment_name": config.experiment_name,
        "config": {
            "model": asdict(config.model),
            "training": asdict(config.training),
            "data": asdict(config.data),
            "evaluation": asdict(config.evaluation)
        },
        "results": metrics,
        "device_info": get_device_info()
    }
    
    summary_path = os.path.join(experiment_dir, "results", "summary.yaml")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)


# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    "rgb_baseline": create_rgb_only_config(),
    "rgbd_full": create_rgbd_config(),
    "rgbd_reduced": create_reduced_layers_config(["layer2", "layer3"]),
}


def load_experiment_config(config_name: str) -> ExperimentConfig:
    """Load a predefined experiment configuration"""
    if config_name in EXPERIMENT_CONFIGS:
        return EXPERIMENT_CONFIGS[config_name]
    else:
        raise ValueError(f"Unknown config: {config_name}. "
                        f"Available: {list(EXPERIMENT_CONFIGS.keys())}")
