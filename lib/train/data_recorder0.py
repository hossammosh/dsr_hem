#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import numpy as np
import threading
import time
import h5py
import glob
# --- Configuration ---
select_sampling = False
# --- Global State (Protected by Lock) ---
_buffer = []
# New matrices to store loss and IoU values
# Rows: samples, Columns: epochs
_loss_matrix = None
_iou_matrix = None
# These will be set in set_epoch based on phase_manager
_max_epochs = None
_max_samples = None

_total_samples_logged_this_epoch = 0
current_epoch = None
_file_lock = threading.RLock()
sample_per_epoch = 0
selected_sampling_epoch = 0

# Hardness calculation parameters
ALPHA = 0.7  # Weight for loss in hardness calculation
BETA = 0.3   # Weight for IoU in hardness calculation
# Define headers based on the original structure
_headers = [
    "Index", "Sample Index", "stats/Loss_total", "stats_IoU", "Seq Name",
    "Template Frame ID", "Template Frame Path", "Search Frame ID", "Seq ID",
    "Seq Path", "Class Name", "Vid ID", "Search Names", "Search Path"
]

# --- Helper Functions ---
def _initialize_matrices(rows, cols):
    """Initialize or resize the loss and IoU matrices with the given dimensions."""
    global _loss_matrix, _iou_matrix
    if _loss_matrix is None or _loss_matrix.shape != (rows, cols):
        _loss_matrix = np.zeros((rows, cols))
    if _iou_matrix is None or _iou_matrix.shape != (rows, cols):
        _iou_matrix = np.zeros((rows, cols))

# --- Filename Generation ---
#def _get_final_filename(settings):
def _get_filename(settings):
    pm=settings.phase_manager
    """Generate filename in the format: phase_{phase}_epoch_{epoch}_samples_{samples}.csv"""
    phase = pm.number
    samples = pm.SPE
    return f'phase_{phase}_epoch_{settings.epoch}_samples_{samples}.csv'
def _get_tmp_filename(settings,temp):
    pm=settings.phase_manager
    """Generate filename in the format: phase_{phase}_epoch_{epoch}_samples_{samples}.csv"""
    phase = pm.number if hasattr(settings, 'phase') else 'train'
    samples = pm.SPE
    return f'phase_{phase}_{temp}_epoch_{settings.epoch}_samples_{samples}.csv'
# def _get_previous_filename(settings):
#     """Generate previous phase's filename in the format: phase_{phase}_epoch_{epoch}_samples_{samples}.csv"""
#     phase = settings.phase if hasattr(settings, 'phase') else 'train'
#     samples = settings.sample_per_epoch if hasattr(settings, 'sample_per_epoch') else 0
#     if settings.epoch > 1:
#         return f'phase_{phase}_epoch_{settings.epoch-1}_samples_{samples}.csv'
#     return None

def _clean_previous_experiments():
    """Cleans up previous experiment files."""
    print("Cleaning up previous experiment files...", flush=True)
    
    # Define file patterns to search for
    file_pattern = "phase_*_epoch_*_samples_*.csv"
    existing_files = glob.glob(file_pattern)
    
    # Print the list of existing files
    if existing_files:
        print(f"Found {len(existing_files)} existing files to clean up:")
        for file in existing_files:
            print(f"  - {file}")
            
        # Delete the files
        for file in existing_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}", flush=True)
            except Exception as e:
                print(f"Error deleting {file}: {e}", flush=True)
    else:
        print("No existing files to clean up.", flush=True)

def set_sampling(ss):
    global select_sampling
    select_sampling = ss

def set_epoch(settings):
    """
    Sets the current epoch, clearing buffers and state for the new epoch.
    If settings.epoch is 1, also cleans up any previous experiment files.
    """
    global _buffer, _total_samples_logged_this_epoch, current_epoch, sample_per_epoch
    global selected_sampling_epoch, _max_epochs, _max_samples
    
    with _file_lock:
        _buffer = []
        current_epoch = settings.epoch
        sample_per_epoch = settings.sample_per_epoch
        selected_sampling_epoch = settings.selected_sampling_epoch
        
        # Initialize matrices based on phase_manager values
        if settings.phase_manager:
            _max_epochs = settings.phase_manager.L1
            _max_samples = settings.phase_manager.SPE1
        else:
            _max_epochs = 5  # Default values if phase_manager is not available
            _max_samples = 1000
            
        # Initialize or update the matrices
        _initialize_matrices(_max_samples, _max_epochs)
        
        if settings.epoch == 1:
            _clean_previous_experiments()

def _calculate_hardness(losses, ious, metadata):
    """
    Calculate hardness scores for all samples in the phase.
    
    Args:
        losses: List of all loss values in the phase
        ious: List of all IoU values in the phase
        metadata: List of metadata dicts for each sample
        
    Returns:
        DataFrame with hardness scores and metadata
    """
    if not losses or not ious or not metadata:
        return pd.DataFrame()
    
    # Create DataFrame from metadata
    df = pd.DataFrame(metadata)
    df['loss'] = losses
    df['iou'] = ious
    
    # 1. Aggregate per sample (average the epoch values)
    sample_metrics = df.groupby('sample_index').agg({
        'loss': 'mean',
        'iou': 'mean',
        'seq_name': 'first',
        'template_frame': 'first',
        'search_frame': 'first'
    }).reset_index()
    
    # 2. Min-max normalization with percentile clipping (1st and 99th percentiles)
    def normalize_with_clipping(series, lower=1, upper=99):
        l_min = np.percentile(series, lower)
        l_max = np.percentile(series, upper)
        # Clip values to [l_min, l_max] range
        clipped = np.clip(series, l_min, l_max)
        # Normalize to [0, 1]
        return (clipped - l_min) / (l_max - l_min + 1e-10)
    
    # Apply normalization
    sample_metrics['loss_norm'] = normalize_with_clipping(sample_metrics['loss'])
    sample_metrics['iou_norm'] = normalize_with_clipping(sample_metrics['iou'])
    
    # 3. Calculate hardness score (weighted sum of normalized loss and 1-IoU)
    sample_metrics['hardness'] = (
        ALPHA * sample_metrics['loss_norm'] + 
        BETA * (1 - sample_metrics['iou_norm'])
    )
    
    # Sort by hardness in descending order
    sample_metrics = sample_metrics.sort_values('hardness', ascending=False)
    
    return sample_metrics

def save_samples(settings):
    """Saves all collected samples and processes phase metrics."""
    global _buffer, _loss_matrix, _iou_matrix
    
    if not _buffer:
        print("No data to save.", flush=True)
        return
        
    try:
        # Create DataFrame from the buffer
        df = pd.DataFrame(_buffer, columns=_headers)
        
        # Save the main CSV file
        filename = _get_filename(settings)
        print(f"Saving {len(df)} samples to {filename}...", flush=True)
        df.to_csv(filename, index=False)
        print(f"Successfully saved {len(df)} samples to {filename}", flush=True)
        # Get the number of epochs and samples per epoch from phase manager
        num_epochs = settings.phase_manager.L
        samples_per_epoch = settings.phase_manager.SPE

        # Create empty matrix with shape (num_epochs, samples_per_epoch)
        _loss_matrix = np.zeros((num_epochs, samples_per_epoch))
        _iou_matrix = np.zeros((num_epochs, samples_per_epoch))

        # Convert buffer to DataFrame if not already done
        if not isinstance(_buffer, pd.DataFrame):
            df = pd.DataFrame(_buffer)
        else:
            df = _buffer

        # Calculate positions in the matrix
        sample_indices = df['Sample Index'].values - 1  # Convert to 0-based
        epoch_indices = (sample_indices // samples_per_epoch).astype(int)
        sample_in_epoch = (sample_indices % samples_per_epoch).astype(int)

        # Filter valid indices
        valid = (epoch_indices >= 0) & (epoch_indices < num_epochs) & \
                (sample_in_epoch >= 0) & (sample_in_epoch < samples_per_epoch)
        
        # Fill matrices using vectorized operations
        _loss_matrix[epoch_indices[valid], sample_in_epoch[valid]] = df['stats/Loss_total'].values[valid]
        _iou_matrix[epoch_indices[valid], sample_in_epoch[valid]] = df['stats_IoU'].values[valid]

        # Store metrics for hardness calculation
        for idx, row in df.iterrows():
            sample_idx = _total_samples_logged_this_epoch + idx
            _loss_matrix[current_epoch - 1, sample_idx] = row['stats/Loss_total']
            _iou_matrix[current_epoch - 1, sample_idx] = row['stats_IoU']
        
        # If this is the last epoch of the phase, process and save hardness metrics
        if settings.phase_manager and settings.epoch == settings.phase_manager.Hepoch:
            print(f"Processing phase {settings.phase_manager.number} metrics...", flush=True)
            
            # Calculate hardness for all samples in the phase
            phase_df = _calculate_hardness(_loss_matrix[:, :current_epoch].mean(axis=1), _iou_matrix[:, :current_epoch].mean(axis=1), df)
            
            if not phase_df.empty:
                # Sort by hardness (descending) and select top SPE2 samples
                phase_df = phase_df.sort_values('hardness', ascending=False)
                top_samples = phase_df.head(settings.phase_manager.SPE2)
                
                # Save all metrics with hardness
                metrics_file = f"phase_{settings.phase_manager.number}_metrics.csv"
                phase_df.to_csv(metrics_file, index=False)
                print(f"Saved phase metrics to {metrics_file}", flush=True)
                
                # Save top hard samples
                hard_samples_file = f"phase_{settings.phase_manager.number}_hard_samples.csv"
                top_samples.to_csv(hard_samples_file, index=False)
                print(f"Saved top {len(top_samples)} hard samples to {hard_samples_file}", flush=True)
        
        # Save a sorted copy of the current epoch's data
        # if len(df) > 0:
        #     filename_tmp = _get_tmp_filename(settings, "tmp")
        #     df_sorted = df.sort_values(by=['stats/Loss_total', 'stats_IoU'],
        #                              ascending=[True, False])
        #     df_sorted.to_csv(filename_tmp, index=False)
        #     print(f"Saved sorted copy to {filename_tmp}", flush=True)
        
        # Clear the buffer after saving
        _buffer = []
        
    except Exception as e:
        print(f"Error saving samples: {e}", flush=True)

def samples_stats_save(sample_index: int, data_info: dict, stats: dict, settings):
    """
    Save sample statistics to the buffer for later logging to CSV.

    Args:
        sample_index: Index of the current sample
        data_info: Dictionary containing sample information
        stats: Dictionary containing sample statistics
    """
    global _buffer, _total_samples_logged_this_epoch
    
    with _file_lock:
        # Create a sample entry with all required fields
        sample_entry = {
            'Index': _total_samples_logged_this_epoch + 1,  # Changed to start from 1
            'Sample Index': sample_index,
            'stats/Loss_total': float(stats.get('Loss/total', 0.0)),
            'stats_IoU': float(stats.get('IoU', 0.0)),
            'Seq Name': _safe_str_list(data_info.get('seq_name', '')),
            'Template Frame ID': _safe_str_list(data_info.get('template', {}).get('frame_id', '')),
            'Template Frame Path': _safe_str_list(data_info.get('template', {}).get('frame_path', '')),
            'Search Frame ID': _safe_str_list(data_info.get('search', {}).get('frame_id', '')),
            'Seq ID': _safe_str_list(data_info.get('seq_id', '')),
            'Seq Path': _safe_str_list(data_info.get('seq_path', '')),
            'Class Name': _safe_str_list(data_info.get('class_name', '')),
            'Vid ID': _safe_str_list(data_info.get('vid_id', '')),
            'Search Names': _safe_str_list(data_info.get('search', {}).get('frame_name', '')),
            'Search Path': _safe_str_list(data_info.get('search', {}).get('frame_path', ''))
        }
        
        _buffer.append(sample_entry)
        _total_samples_logged_this_epoch += 1
        
        # If we've collected all samples for this epoch, save them
        if _total_samples_logged_this_epoch == settings.phase_manager.SPE:
            save_samples(settings)


def _safe_str_list(value):
    """Safely convert lists or other types to string."""
    if isinstance(value, list):
        return ", ".join(map(str, value))
    elif value is None:
        return ""
    else:
        return str(value)

def save_gradients(model, sample_index, epoch, output_dir='gradients'):
    """
    Save model gradients to an HDF5 file.

    Args:
        model: The PyTorch model
        sample_index: Index of the current sample
        epoch: Current epoch number
        output_dir: Directory to save gradient files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        gradients = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients.append({
                    'name': name,
                    'grad': param.grad.cpu().numpy(),
                    'shape': list(param.grad.shape)
                })
        
        if gradients:
            filename = os.path.join(output_dir, f'gradients_epoch_{epoch}_sample_{sample_index}.h5')
            with h5py.File(filename, 'w') as f:
                for i, grad in enumerate(gradients):
                    grp = f.create_group(f'grad_{i}')
                    grp.attrs['name'] = grad['name']
                    grp.attrs['shape'] = grad['shape']
                    grp.create_dataset('gradient', data=grad['grad'])
            
            print(f"Saved gradients to {filename}", flush=True)
            
    except Exception as e:
        print(f"Error saving gradients: {e}", flush=True)