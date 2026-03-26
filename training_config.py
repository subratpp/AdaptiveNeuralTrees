"""
Dataset-specific training and hyperparameter configurations.
These parameters are recommended defaults for each dataset.
They can be overridden via command-line arguments to tree.py.
"""

TRAINING_CONFIGS = {
    "mnist": {
        "learning_rate": 0.0001,
        "batch_size": 256,
        "use_gpu": True,
        "epochs_node": 20,
        "epochs_finetune": 20,
        "epochs_patience": 5,
        "maxdepth": 8,
        "router_ver": 2,
        "router_ngf": 64,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 2,
        "transformer_ngf": 64,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 3,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "plateau",
        "valid_ratio": 0.1,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": False,
        "finetune_during_growth": False,
    },
    "cifar10": {
        "learning_rate": 0.001,
        "batch_size": 512,
        "use_gpu": True,
        "epochs_node": 100,
        "epochs_finetune": 200,
        "epochs_patience": 5,
        "maxdepth": 10,
        "router_ver": 3,
        "router_ngf": 128,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 5,
        "transformer_ngf": 128,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 6,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "step_lr",
        "valid_ratio": 0.1,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": True,
        "finetune_during_growth": False,
    },
    "letter": {
        "learning_rate": 0.005,
        "batch_size": 128,
        "use_gpu": True,
        "epochs_node": 100,
        "epochs_finetune": 100,
        "epochs_patience": 5,
        "maxdepth": 10,
        "router_ver": 4,
        "router_ngf": 1,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 1,
        "transformer_ngf": 3,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 3,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "plateau",
        "valid_ratio": 0.2,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": False,
        "finetune_during_growth": False,
    },
    "connect": {
        "learning_rate": 0.001,
        "batch_size": 128,
        "use_gpu": True,
        "epochs_node": 20,
        "epochs_finetune": 20,
        "epochs_patience": 5,
        "maxdepth": 8,
        "router_ver": 4,
        "router_ngf": 1,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 1,
        "transformer_ngf": 3,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 3,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "plateau",
        "valid_ratio": 0.2,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": False,
        "finetune_during_growth": False,
    },
    "census": {
        "learning_rate": 0.001,
        "batch_size": 256,
        "use_gpu": True,
        "epochs_node": 80,
        "epochs_finetune": 150,
        "epochs_patience": 5,
        "maxdepth": 8,
        "router_ver": 4,
        "router_ngf": 1,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 1,
        "transformer_ngf": 3,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 3,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "plateau",
        "valid_ratio": 0.2,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": False,
        "finetune_during_growth": False,
    },
    "forest": {
        "learning_rate": 0.01,
        "batch_size": 128,
        "use_gpu": True,
        "epochs_node": 100,
        "epochs_finetune": 200,
        "epochs_patience": 5,
        "maxdepth": 10,
        "router_ver": 4,
        "router_ngf": 1,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 1,
        "transformer_ngf": 3,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 3,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "plateau",
        "valid_ratio": 0.2,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": False,
        "finetune_during_growth": False,
    },
    "segment": {
        "learning_rate": 0.005,
        "batch_size": 128,
        "use_gpu": True,
        "epochs_node": 50,
        "epochs_finetune": 30,
        "epochs_patience": 5,
        "maxdepth": 8,
        "router_ver": 4,
        "router_ngf": 1,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 1,
        "transformer_ngf": 3,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 3,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "plateau",
        "valid_ratio": 0.2,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": False,
        "finetune_during_growth": False,
    },
    "satimages": {
        "learning_rate": 0.005,
        "batch_size": 128,
        "use_gpu": True,
        "epochs_node": 15,
        "epochs_finetune": 5,
        "epochs_patience": 5,
        "maxdepth": 6,
        "router_ver": 4,
        "router_ngf": 1,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 1,
        "transformer_ngf": 3,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 3,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "plateau",
        "valid_ratio": 0.2,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": False,
        "finetune_during_growth": False,
    },
    "pendigits": {
        "learning_rate": 0.005,
        "batch_size": 128,
        "use_gpu": True,
        "epochs_node": 50,
        "epochs_finetune": 30,
        "epochs_patience": 5,
        "maxdepth": 8,
        "router_ver": 4,
        "router_ngf": 1,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 1,
        "transformer_ngf": 3,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 3,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "plateau",
        "valid_ratio": 0.2,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": False,
        "finetune_during_growth": False,
    },
    "protein": {
        "learning_rate": 0.0001,
        "batch_size": 128,
        "use_gpu": True,
        "epochs_node": 10,
        "epochs_finetune": 5,
        "epochs_patience": 5,
        "maxdepth": 4,
        "router_ver": 4,
        "router_ngf": 1,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 1,
        "transformer_ngf": 3,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 3,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "plateau",
        "valid_ratio": 0.2,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": False,
        "finetune_during_growth": False,
    },
    "sensit": {
        "learning_rate": 0.001,
        "batch_size": 128,
        "use_gpu": True,
        "epochs_node": 100,
        "epochs_finetune": 100,
        "epochs_patience": 5,
        "maxdepth": 10,
        "router_ver": 4,
        "router_ngf": 1,
        "router_k": 28,
        "router_dropout_prob": 0.0,
        "transformer_ver": 1,
        "transformer_ngf": 3,
        "transformer_k": 5,
        "transformer_expansion_rate": 1,
        "transformer_reduction_rate": 2,
        "solver_ver": 3,
        "solver_dropout_prob": 0.0,
        "solver_inherit": False,
        "scheduler": "plateau",
        "valid_ratio": 0.2,
        "downsample_interval": 0,
        "visualization_split": False,
        "batch_norm": False,
        "finetune_during_growth": False,
    },
}


def get_training_config(dataset):
    """Get recommended training configuration for a dataset.
    
    Args:
        dataset (str): Dataset code (e.g., 'mnist', 'letter', etc.)
    
    Returns:
        dict: Training configuration with recommended hyperparameters.
    """
    if dataset in TRAINING_CONFIGS:
        return TRAINING_CONFIGS[dataset].copy()
    else:
        # Return a generic safe default for unknown datasets
        return {
            "learning_rate": 0.001,
            "batch_size": 128,
            "use_gpu": True,
            "epochs_node": 50,
            "epochs_finetune": 100,
            "epochs_patience": 5,
            "maxdepth": 8,
            "router_ver": 1,
            "router_ngf": 64,
            "router_k": 28,
            "router_dropout_prob": 0.0,
            "transformer_ver": 1,
            "transformer_ngf": 3,
            "transformer_k": 5,
            "transformer_expansion_rate": 1,
            "transformer_reduction_rate": 2,
            "solver_ver": 1,
            "solver_dropout_prob": 0.0,
            "solver_inherit": False,
            "scheduler": "plateau",
            "valid_ratio": 0.1,
            "downsample_interval": 0,
            "visualization_split": False,
            "batch_norm": False,
            "finetune_during_growth": False,
        }


def apply_training_config_to_args(args, cli_args=None):
    """Apply training config from TRAINING_CONFIGS to args object.
    
    This function loads the dataset-specific training configuration and applies it
    to the argparse args object. Command-line arguments take precedence over
    config defaults.
    
    Args:
        args: argparse.Namespace object with parsed command-line arguments
    
    Returns:
        args: Updated argparse.Namespace object with training config applied
    """
    config = get_training_config(args.dataset)
    cli_args = cli_args or []

    # Map config keys to command-line flags so explicit user flags are preserved.
    cli_flag_map = {
        'learning_rate': ['--lr'],
        'batch_size': ['--batch-size'],
        'use_gpu': ['--use_gpu', '--no_cuda'],
        'epochs_node': ['--epochs_node'],
        'epochs_finetune': ['--epochs_finetune'],
        'epochs_patience': ['--epochs_patience'],
        'maxdepth': ['--maxdepth'],
        'router_ver': ['--router_ver', '-r_ver'],
        'router_ngf': ['--router_ngf', '-r_ngf'],
        'router_k': ['--router_k', '-r_k'],
        'router_dropout_prob': ['--router_dropout_prob', '-r_drop'],
        'transformer_ver': ['--transformer_ver', '-t_ver'],
        'transformer_ngf': ['--transformer_ngf', '-t_ngf'],
        'transformer_k': ['--transformer_k', '-t_k'],
        'transformer_expansion_rate': ['--transformer_expansion_rate', '-t_expr'],
        'transformer_reduction_rate': ['--transformer_reduction_rate', '-t_redr'],
        'solver_ver': ['--solver_ver', '-s_ver'],
        'solver_dropout_prob': ['--solver_dropout_prob', '-s_drop'],
        'solver_inherit': ['--solver_inherit', '-s_inh'],
        'scheduler': ['--scheduler'],
        'valid_ratio': ['--valid_ratio', '-vr'],
        'downsample_interval': ['--downsample_interval', '-ds_int'],
        'visualization_split': ['--visualise_split'],
        'batch_norm': ['--batch_norm', '-bn'],
        'finetune_during_growth': ['--finetune_during_growth'],
    }

    def was_explicitly_set(flags):
        for cli in cli_args:
            for flag in flags:
                if cli == flag or cli.startswith(flag + '='):
                    return True
        return False
    
    # List of config parameters to apply to args
    config_keys = [
        'learning_rate', 'batch_size', 'use_gpu', 'epochs_node', 'epochs_finetune', 
        'epochs_patience', 'maxdepth', 'router_ver', 'router_ngf', 'router_k',
        'router_dropout_prob', 'transformer_ver', 'transformer_ngf', 'transformer_k',
        'transformer_expansion_rate', 'transformer_reduction_rate', 
        'solver_ver', 'solver_dropout_prob', 'solver_inherit', 'scheduler', 
        'valid_ratio', 'downsample_interval', 'visualization_split', 
        'batch_norm', 'finetune_during_growth'
    ]
    
    for key in config_keys:
        if key in config:
            if was_explicitly_set(cli_flag_map.get(key, [])):
                continue
            # Map visualization_split to visualise_split for consistency with tree.py
            if key == 'visualization_split':
                setattr(args, 'visualise_split', config[key])
            else:
                setattr(args, key, config[key])
    
    # Map 'lr' from training_config to actual 'lr' attribute
    if hasattr(args, 'learning_rate') and not was_explicitly_set(['--lr']):
        args.lr = args.learning_rate
    
    return args
