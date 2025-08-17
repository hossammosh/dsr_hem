#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import numpy as np
import threading
import glob
import torch
import random
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
_file_lock = threading.RLock()
sample_per_epoch = 0

# Hardness calculation parameters
ALPHA = 0.7  # Weight for loss in hardness calculation
BETA = 0.3  # Weight for IoU in hardness calculation
# Define headers based on the original structure
_headers = [
    "Index", "Sample Index", "stats/Loss_total", "stats_IoU", "Seq Name",
    "Template Frame ID", "Template Frame Path", "Search Frame ID", "Seq ID",
    "Seq Path", "Class Name", "Vid ID", "Search Names", "Search Path", "Hardness_Score"
]

# --- Filename Generation ---
def _get_filename(settings):
    pm = settings.phase_manager
    """Generate filename in the format: phase_{phase}_epoch_{epoch}_samples_{samples}.csv"""
    phase = pm.number
    samples = pm.SPE
    return f'phase_{phase}_epoch_{settings.epoch}_samples_{samples}.csv'

def _get_tmp_filename(settings, temp):
    return f'{temp}_epoch_{settings.epoch+1}.csv'

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
    global _buffer, _total_samples_logged_this_epoch

    with _file_lock:
        # Reset for new epoch
        _buffer = []
        _total_samples_logged_this_epoch = 0

        if settings.epoch == 1:
            _clean_previous_experiments()
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

        loss2d = np.array([[d['stats/Loss_total']] for d in _buffer])
        _loss_matrix.append(loss2d)
        loss2d=[]
        iou2d = np.array([[d["stats_IoU"]] for d in _buffer])
        _iou_matrix.append(iou2d)
        iou2d=[]
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
    global _buffer, _total_samples_logged_this_epoch

    with _file_lock:
        # Create a sample entry with all required fields
        sample_entry = {
            'Index': _total_samples_logged_this_epoch + 1,  # Changed to start from 1
            'Sample Index': sample_index,
            'stats/Loss_total': float(stats.get('Loss/total', 0.0)),
            'stats_IoU': float(stats.get('IoU', 0.0)),

            # Direct keys
            'Seq Name': _safe_str_list(data_info.get('seq_name', '')),
            'Seq ID': _safe_str_list(data_info.get('seq_id', '')),
            'Seq Path': _safe_str_list(data_info.get('seq_path', '')),
            'Class Name': _safe_str_list(data_info.get('class_name', '')),
            'Vid ID': _safe_str_list(data_info.get('vid_id', '')),

            # Template info (flat keys)
            'Template Frame ID': _safe_str_list(data_info.get('template_ids', [])),
            'Template Frame Path': _safe_str_list(data_info.get('template_path', [])),
            'Template Frame Name': _safe_str_list(data_info.get('template_names', [])),

            # Search info (flat keys)
            'Search Frame ID': _safe_str_list(data_info.get('search_id', [])),
            'Search Path': _safe_str_list(data_info.get('search_path', [])),
            'Search Names': _safe_str_list(data_info.get('search_names', [])),
        }

        _buffer.append(sample_entry)
        _total_samples_logged_this_epoch += 1

        # If we've collected all samples for this epoch, save them
        if _total_samples_logged_this_epoch == settings.phase_manager.SPE:
            save_samples(settings)
            # Check if we've reached the end of the current phase
            if hasattr(settings, 'phase_manager') and settings.epoch == settings.phase_manager.L:
                # Check if we're in phase 1
                if settings.phase_manager.number == 1:
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
                    avg_ious = np.mean(ious_3d[:, :, 0], axis=0)  # Shape: (samples,)
                    # Calculate 1st and 99th percentiles for IoUs
                    Imin = np.percentile(avg_ious, 1)
                    Imax = np.percentile(avg_ious, 99)
                    avg_ious_tensor = torch.from_numpy(avg_ious).float()
                    clipped_ious = torch.clamp(avg_ious_tensor, min=Imin, max=Imax).numpy()
                    ious_norm = (clipped_ious - Imin) / (Imax - Imin)
                    
                    # Calculate hardness scores using the formula: hardness = alpha * loss_norm + beta * (1 - iou_norm)
                    alpha = 0.7  # weight for loss component
                    beta = 0.3   # weight for IoU component
                    hardness_scores = alpha * loss_norm + beta * (1 - ious_norm)
                    
                    # Store the hardness scores in the buffer entries
                    for i, entry in enumerate(_buffer):
                        if i < len(hardness_scores):
                            entry['Hardness_Score'] = float(hardness_scores[i])
                    
                    # Sort buffer by Hardness_Score in descending order (hardest first)
                    _buffer.sort(key=lambda x: x.get('Hardness_Score', 0), reverse=True)
                    num_samples_to_keep = int(len(_buffer) * settings.phase_manager.SPE2_ratio)
                    _buffer_copy=_buffer
                    _buffer = _buffer[:num_samples_to_keep]

                    output_file = _get_tmp_filename(settings, 'source_phase2')
                    # Create DataFrame and save to CSV
                    df = pd.DataFrame(_buffer)
                    settings.phase_manager.ds_phase2 =df
                    df.to_csv(output_file, index=False)
                    excel_file = output_file.replace('.csv', '.xlsx')
                    df.to_excel(excel_file, index=False)
                    print(f"Saved {len(df)} cropped samples to {output_file} and {excel_file}", flush=True)

                    output_file = _get_tmp_filename(settings, 'low hardening samples 40%')
                    dslh_samples = _buffer_copy[num_samples_to_keep:]  # Get samples after num_samples_to_keep
                    dslh_samples = pd.DataFrame(dslh_samples)
                    dslh_samples.to_csv(output_file, index=False)
                    excel_file = output_file.replace('.csv', '.xlsx')
                    dslh_samples.to_excel(excel_file, index=False)
                    print(f"Saved {len(dslh_samples)} cropped samples to {output_file} and {excel_file}", flush=True)

                    diversity_samples = settings.phase_manager.DSLH
                    dslh_ss_indices = np.random.randint(0, len(dslh_samples), size=diversity_samples).tolist()
                    settings.phase_manager.dslh_ss = dslh_samples.loc[dslh_ss_indices]
                    print(f"Selected {len(settings.phase_manager.dslh_ss)} random samples from ds_low_hardness_samples (DSLH={settings.phase_manager.DSLH})")
                    output_file_dslh_ss = _get_tmp_filename(settings, 'data set low hardness diversity samples')
                    dslh_ss = pd.DataFrame(settings.phase_manager.dslh_ss)
                    dslh_ss.to_csv(output_file_dslh_ss, index=False)
                    excel_file = output_file_dslh_ss.replace('.csv', '.xlsx')
                    dslh_ss.to_excel(excel_file, index=False)

                    combined_dslh = pd.concat([df, dslh_ss], ignore_index=True)
                    settings.phase_manager.ds_phase4 = combined_dslh
                    output_file_combined = _get_tmp_filename(settings, 'combined_data_set_source_phase4')
                    combined_dslh.to_csv(output_file_combined, index=False)
                    excel_file_combined = output_file_combined.replace('.csv', '.xlsx')
                    combined_dslh.to_excel(excel_file_combined, index=False)
                    print(
                        f"Saved {len(combined_dslh)} combined DSLH samples to {output_file_combined} and {excel_file_combined}",
                        flush=True)


def _safe_str_list(value):
    """Safely convert lists or other types to string."""
    if isinstance(value, list):
        return ", ".join(map(str, value))
    elif value is None:
        return ""
    else:
        return str(value)
