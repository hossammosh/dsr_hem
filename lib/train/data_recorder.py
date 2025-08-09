#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import numpy as np
import threading
import glob
import torch
# --- Configuration ---
select_sampling = False
# --- Global State (Protected by Lock) ---
_buffer = []
# Rows: samples, Columns: epochs
_loss_matrix = []
_iou_matrix = []
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
BETA = 0.3  # Weight for IoU in hardness calculation
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
    pass


# --- Filename Generation ---
# def _get_final_filename(settings):
def _get_filename(settings):
    pm = settings.phase_manager
    """Generate filename in the format: phase_{phase}_epoch_{epoch}_samples_{samples}.csv"""
    phase = pm.number
    samples = pm.SPE
    return f'phase_{phase}_epoch_{settings.epoch}_samples_{samples}.csv'


def _get_tmp_filename(settings, temp):
    pm = settings.phase_manager
    """Generate filename in the format: phase_{phase}_epoch_{epoch}_samples_{samples}.csv"""
    phase = pm.number if hasattr(settings, 'phase') else 'train'
    samples = pm.SPE
    return f'phase_{phase}_{temp}_epoch_{settings.epoch}_samples_{samples}.csv'

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

def set_epoch(settings):
    """
    Sets the current epoch, clearing buffers and state for the new epoch.
    If settings.epoch is 1, also cleans up any previous experiment files.
    """
    global _buffer, _total_samples_logged_this_epoch, current_epoch, sample_per_epoch
    global selected_sampling_epoch

    with _file_lock:
        # Save any remaining samples from the previous epoch
        if _buffer:
            save_samples(settings)
            
        # Reset for new epoch
        _buffer = []
        _total_samples_logged_this_epoch = 0
        current_epoch = settings.epoch
        sample_per_epoch = settings.sample_per_epoch
        selected_sampling_epoch = settings.selected_sampling_epoch

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

        stat2d = np.array([[d['stats/Loss_total']] for d in _buffer])
        _loss_matrix.append(stat2d)
        iou2d = np.array([[d["stats_IoU"]] for d in _buffer])
        _iou_matrix.append(iou2d)
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
        settings: Training settings containing phase information
    """
    global _buffer, _total_samples_logged_this_epoch, _loss_matrix

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
            # Check if we've reached the end of the current phase
            if hasattr(settings, 'phase_manager') and settings.epoch == settings.phase_manager.L:
                # Check if we're in phase 1
                if settings.phase_manager.number == 1 and _loss_matrix:
                    # Convert _loss_matrix to 3D numpy array (epochs x samples x 2 for loss and IoU)
                    loss_3d = np.array(_loss_matrix)  # Shape: (epochs, samples, 2)
                    # Calculate average loss across epochs for each sample
                    avg_losses = np.mean(loss_3d[:, :, 0], axis=0)  # Shape: (samples,)
                    # Calculate 1st and 99th percentiles for losses
                    Lmin = np.percentile(avg_losses, 1)
                    Lmax = np.percentile(avg_losses, 99)
                    avg_losses_tensor = torch.from_numpy(avg_losses).float()
                    # Now use torch.clamp
                    clipped_loss = torch.clamp(avg_losses_tensor, min=Lmin, max=Lmax).numpy()
                    loss_norm = (clipped_loss - Lmin) / (Lmax - Lmin)

                    ious_3d = np.array(_iou_matrix)
                    # Calculate average IoU across epochs for each sample
                    avg_ious = np.mean(loss_3d[:, :, 1], axis=0)  # Shape: (samples,)
                    # Calculate 1st and 99th percentiles for IoUs
                    Imin = np.percentile(avg_ious, 1)
                    Imax = np.percentile(avg_ious, 99)
                    avg_ious_tensor = torch.from_numpy(avg_ious).float()
                    clipped_ious = torch.clamp(avg_ious_tensor, min=Imin, max=Imax).numpy()
                    ious_norm = (clipped_ious - Imin) / (Imax - Imin)


def _safe_str_list(value):
    """Safely convert lists or other types to string."""
    if isinstance(value, list):
        return ", ".join(map(str, value))
    elif value is None:
        return ""
    else:
        return str(value)
