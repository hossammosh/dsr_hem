"""
Phase class for managing training phases in the SeqTrack model.
This module handles phase management and automatically loads configuration from YAML.
"""
import yaml
import os
from pathlib import Path

class Phase:
    """
    A class to manage training phases with their properties and methods.
    Automatically loads phase constants from the YAML configuration file.
    
    Attributes:
        config_path (str): Path to the YAML configuration file
        number (int): Current phase number (1-4)
        name (str): Name of the current phase
        Lepoch (int): Lower epoch bound of the phase
        Hepoch (int): Upper epoch bound of the phase
        SPE (int): Samples per epoch for this phase
        DESC (str): Description of the phase
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Phase class.
        
        Args:
            config_path (str, optional): Path to the YAML config file. If None, will try to find it.
        """
        self.number = 0      # Current phase number (1-4)
        self.name = ""       # Name of the current phase
        self.Lepoch = 0      # Lower epoch bound of the phase
        self.Hepoch = 0      # Upper epoch bound of the phase
        self.SPE = 0         # Samples per epoch for this phase
        self.DESC = ""       # Description of the phase
        self.L = 0
        
        # Set default config path if not provided
        self.config_path = config_path
    
    def _load_phase_constants(self):
        """Load phase constants from the YAML configuration file and store them as class attributes."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Get base samples per epoch
        self.base_samples = config['DATA']['TRAIN'].get('SAMPLE_PER_EPOCH', 1)
        # Extract phase settings
        phases = config.get('PHASES', {})
        
        # Store phase constants as class attributes
        self.L1 = phases.get('L1', 5)
        self.L2 = phases.get('L2', 10)
        self.L3 = phases.get('L3', 15)
        self.L4 = phases.get('L4', 20)
        
        # Store SPE ratios and calculate actual sample counts
        self.SPE1_ratio = phases.get('SPE1', 1.0)
        self.SPE2_ratio = phases.get('SPE2', 0.6)
        self.SPE3_ratio = phases.get('SPE3', 0.6)
        self.SPE4_ratio = phases.get('SPE4', 0.25)
        
        # Calculate actual sample counts
        self.SPE1 = int(self.SPE1_ratio * self.base_samples)
        self.SPE2 = int(self.SPE2_ratio * self.base_samples)
        self.SPE3 = int(self.SPE3_ratio * self.base_samples)
        self.SPE4 = int(self.SPE4_ratio * self.base_samples)
        
        self.total_epochs = self.base_samples  # Use SAMPLE_PER_EPOCH as the default total number of epochs

    def set_phase(self, epoch):
        """
        Set the phase based on the current epoch and configuration.
        
        Args:
            epoch (int): Current training epoch
        """
        # Load phase constants from config
        self._load_phase_constants()
        
        # Set phase based on epoch
        if 1 <= epoch <= self.L1:
            self.number = 1
            self.name = "Warm-up Phase"
            self.Lepoch = 1
            self.Hepoch = self.L1
            self.SPE = self.SPE1
            self.DESC = "Train normally + log per-sample loss"
            self.L = self.L1

        elif self.L1 + 1 <= epoch <= self.L2:
            self.number = 2
            self.name = "First HEM Phase"
            self.Lepoch = self.L1 + 1
            self.Hepoch = self.L2
            self.SPE = self.SPE2
            self.DESC = "Train with HEM + log per-sample loss"
            self.L = self.L2
            
        elif self.L2 + 1 <= epoch <= self.L3:
            self.number = 3
            self.name = "Second HEM Phase"
            self.Lepoch = self.L2 + 1
            self.Hepoch = self.L3
            self.SPE = self.SPE3
            self.DESC = "Train with HEM + log per-sample loss"
            self.L = self.L3
            
        elif self.L3 + 1 <= epoch <= self.L4:
            self.number = 4
            self.name = "Final HEM Phase"
            self.Lepoch = self.L3 + 1
            self.Hepoch = self.L4
            self.SPE = self.SPE4
            self.DESC = "Train with HEM + log per-sample loss"
            self.L = self.L4
        else:
            # Default to phase 4 if beyond defined phases
            self.number = 4
            self.name = "Extended Final Phase"
            self.Lepoch = self.L4 + 1
            self.Hepoch = float('inf')
            self.SPE = self.SPE4
            self.DESC = "Continue training with final phase settings"
            self.L = self.L4
