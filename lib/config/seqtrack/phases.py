"""
Phase configurations for different training phases.
This module contains the phase configurations that can be imported and used across the project.
"""
from easydict import EasyDict as edict

def get_phases_config():
    """
    Returns the phases configuration dictionary with lambda functions for dynamic EPOCH_RANGE calculation. 123
    
    Returns:
        dict: Dictionary containing phase configurations with lambda functions for EPOCH_RANGE but it should be called
    """
    return {
        'warmup': {
            'NAME': "Warm-up Phase",
            'EPOCH_RANGE': lambda L1, L2, L3, L4: [1, L1],
            'SAMPLE_PER_EPOCH': "SPE1",
            'DESC': "Train normally + log per-sample loss"
        },
        'first_hem': {
            'NAME': "First HEM Phase",
            'EPOCH_RANGE': lambda L1, L2, L3, L4: [L1 + 1, L2],
            'SAMPLE_PER_EPOCH': "SPE2",
            'DESC': "Train on top 60% hardest samples"
        },
        'remining': {
            'NAME': "Re-mining Phase",
            'EPOCH_RANGE': lambda L1, L2, L3, L4: [L2 + 1, L3],
            'SAMPLE_PER_EPOCH': "SPE3",
            'DESC': "Train + re-log per-sample loss"
        },
        'refined_hem': {
            'NAME': "Refined HEM Phase",
            'EPOCH_RANGE': lambda L1, L2, L3, L4: [L3 + 1, L4],
            'SAMPLE_PER_EPOCH': "SPE4",
            'DESC': "Train on 60% hardest + 10% random samples",
            'HARD_SAMPLES_RATIO': 0.6,
            'RANDOM_SAMPLES_RATIO': 0.1
        }
    }

def get_phase_constants(cfg):
    """
    Returns the phase-related constants from the config.
    
    Args:
        cfg: The configuration object containing phase settings
        
    Returns:
        dict: Dictionary containing phase-related constants
    """
    # Get base samples per epoch from DATA.TRAIN.SAMPLE_PER_EPOCH in the config
    base_samples = cfg.DATA.TRAIN.SAMPLE_PER_EPOCH
    
    return {
        'SPE1': int(base_samples * cfg.PHASES.SPE1),  # Samples per epoch for phase 1
        'SPE2': int(base_samples * cfg.PHASES.SPE2),  # Samples per epoch for phase 2
        'SPE3': int(base_samples * cfg.PHASES.SPE3),  # Samples per epoch for phase 3
        'SPE4': int(base_samples * cfg.PHASES.SPE4),  # Samples per epoch for phase 4
        'L1': cfg.PHASES.L1,       # Epoch boundary 1
        'L2': cfg.PHASES.L2,       # Epoch boundary 2
        'L3': cfg.PHASES.L3,       # Epoch boundary 3
        'L4': cfg.PHASES.L4        # Epoch boundary 4
    }

def initialize_phases(phases_cfg, cfg):
    """
    Initialize the phase configurations in the provided config object.
    
    Args:
        phases_cfg: The PHASES configuration object to initialize
        cfg: The main configuration object containing phase settings
    """
    # Set phase constants from the config
    constants = get_phase_constants(cfg)
    for key, value in constants.items():
        setattr(phases_cfg, key, value)
    
    # Set up phase configurations
    phases_dict = get_phases_config()
    
    # Convert the dictionary to edict and calculate EPOCH_RANGE
    for phase_name, phase_config in phases_dict.items():
        phase_edict = edict(phase_config)
        if 'EPOCH_RANGE' in phase_config and callable(phase_config['EPOCH_RANGE']):
            phase_edict.EPOCH_RANGE = phase_config['EPOCH_RANGE'](
                constants['L1'], constants['L2'], constants['L3'], constants['L4']
            )
        setattr(phases_cfg, phase_name, phase_edict)
