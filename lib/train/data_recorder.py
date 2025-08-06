#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import threading
import time
import h5py
import glob

# --- Configuration ---
select_sampling = False
# --- Global State (Protected by Lock) ---
_buffer = []
_total_samples_logged_this_epoch = 0
current_epoch = None
_file_lock = threading.RLock()
sample_per_epoch = 0
selected_sampling_epoch = 0
# Define headers based on the original structure
_headers = [
    "Index", "Sample Index", "stats/Loss_total", "stats_IoU", "Seq Name",
    "Template Frame ID", "Template Frame Path", "Search Frame ID", "Seq ID",
    "Seq Path", "Class Name", "Vid ID", "Search Names", "Search Path"
]

# --- Filename Generation ---
def _get_final_filename(settings):
    """Generate filename in the format: phase_{phase}_epoch_{epoch}_samples_{samples}.csv"""
    phase = settings.phase if hasattr(settings, 'phase') else 'train'
    samples = settings.sample_per_epoch if hasattr(settings, 'sample_per_epoch') else 0
    return f'phase_{phase}_epoch_{settings.epoch}_samples_{samples}.csv'

def _get_previous_filename(settings):
    """Generate previous phase's filename in the format: phase_{phase}_epoch_{epoch}_samples_{samples}.csv"""
    phase = settings.phase if hasattr(settings, 'phase') else 'train'
    samples = settings.sample_per_epoch if hasattr(settings, 'sample_per_epoch') else 0
    if settings.epoch > 1:
        return f'phase_{phase}_epoch_{settings.epoch-1}_samples_{samples}.csv'
    return None

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
    global _buffer, _total_samples_logged_this_epoch, current_epoch, sample_per_epoch, selected_sampling_epoch
    
    with _file_lock:
        _buffer = []
        _total_samples_logged_this_epoch = 0
        current_epoch = settings.epoch
        sample_per_epoch = settings.sample_per_epoch
        selected_sampling_epoch = settings.selected_sampling_epoch
        
        if settings.epoch == 1:
            _clean_previous_experiments()

def save_samples(settings):
    """Saves all collected samples to a single CSV file."""
    global _buffer
    
    if not _buffer:
        print("No data to save.", flush=True)
        return
        
    try:
        # Create DataFrame from the buffer with the correct column order
        df = pd.DataFrame(_buffer, columns=_headers)
        
        # Get the final filename
        filename = _get_final_filename(settings)
        print(f"Saving {len(df)} samples to {filename}...", flush=True)
        
        # Save as CSV
        df.to_csv(filename, index=False)
        print(f"Successfully saved {len(df)} samples to {filename}", flush=True)
        
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
        if _total_samples_logged_this_epoch >= sample_per_epoch:
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