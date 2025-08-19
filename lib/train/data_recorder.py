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
    file_patterns = [
        "phase_*_epoch_*_samples_*.csv",
        "source_phase2_epoch_*.xlsx",
        "source_phase4_epoch_*.xlsx"
    ]
    existing_files = []
    for pattern in file_patterns:
        existing_files.extend(glob.glob(pattern))

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
def _get_old_filename(settings,epoch):
    pm = settings.phase_manager
    """Generate filename in the format: phase_{phase}_epoch_{epoch}_samples_{samples}.csv"""
    phase = pm.number
    samples = pm.SPE

    return f'phase_{phase}_epoch_{epoch}_samples_{samples}.csv'

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
def set_epoch_from_checkpoint(settings,load_ckpt):
    """
    Sets the current epoch, clearing buffers and state for the new epoch.
    If settings.epoch is 1, also cleans up any previous experiment files.
    """
    global _buffer, _total_samples_logged_this_epoch
    with _file_lock:
        if (settings.phase_manager.number == 1):
            for epoch in range(1, load_ckpt + 1):
                try:
                    # Get the filename for the current epoch's checkpoint
                    filename = _get_old_filename(settings, epoch)
                    if os.path.exists(filename):
                        try:
                            # Read the file based on extension
                            if filename.endswith('.csv'):
                                df = pd.read_csv(filename)
                            elif filename.endswith(('.xlsx', '.xls')):
                                df = pd.read_excel(filename)
                            else:
                                print(f"Unsupported file format for {filename}")
                                continue
                            _buffer.extend(df.to_dict('records'))
                            print(f"Loaded checkpoint'stats for  epoch {epoch}")
                        except Exception as e:
                            print(f"Error loading {filename}: {str(e)}")
                    else:
                        print(f"Warning: Checkpoint for epoch {epoch} not found at {filename}")
                except Exception as e:
                    print(f"Error loading checkpoint for epoch {epoch}: {str(e)}")
                loss2d = np.array([[d['stats/Loss_total']] for d in _buffer])
                _loss_matrix.append(loss2d)
                loss2d = []
                iou2d = np.array([[d["stats_IoU"]] for d in _buffer])
                _iou_matrix.append(iou2d)
                iou2d = []
                _buffer=[]
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
    global _buffer, _total_samples_logged_this_epoch, _loss_matrix, _iou_matrix
    with _file_lock:
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

        if _total_samples_logged_this_epoch == settings.phase_manager.SPE:
            save_samples(settings)
            # Check if we've reached the end of the current phase

            if settings.epoch == settings.phase_manager.L1:
                hardness_scores = calculate_hardness_scores(settings, _loss_matrix, _iou_matrix, alpha=0.7, beta=0.3)
                _loss_matrix = []
                _iou_matrix = []
                for i, entry in enumerate(_buffer):
                    if i < len(hardness_scores):
                        entry['Hardness_Score'] = float(hardness_scores[i])

                _buffer.sort(key=lambda x: x.get('Hardness_Score', 0), reverse=True)
                _buffer_copy = _buffer
                _buffer = _buffer[:settings.phase_manager.SPE2]
                df = pd.DataFrame(_buffer)
                settings.phase_manager.ds_phase2 = df
                settings.phase_manager.dslh = pd.DataFrame(_buffer_copy[settings.phase_manager.SPE2:])
                output_file = _get_tmp_filename(settings, 'source_phase2')
                excel_file = output_file.replace('.csv', '.xlsx')
                df.to_excel(excel_file, index=False)
                output_file = _get_tmp_filename(settings, 'first_stage_low_hardness_samples')
                excel_file = output_file.replace('.csv', '.xlsx')
                settings.phase_manager.dslh.to_excel(excel_file, index=False)
#########  ******************* 3 Phase

            if settings.epoch == settings.phase_manager.L3:
                hardness_scores = calculate_hardness_scores(settings, _loss_matrix, _iou_matrix, alpha=0.7, beta=0.3)
                _loss_matrix = []
                _iou_matrix = []
                for i, entry in enumerate(_buffer):
                    if i < len(hardness_scores):
                        entry['Hardness_Score'] = float(hardness_scores[i])
                diversity_samples = settings.phase_manager.DSLH
                dslh_ss_indices = np.random.randint(0, len(settings.phase_manager.dslh),
                                                    size=diversity_samples).tolist()
                dslh_ss=settings.phase_manager.dslh_ss = settings.phase_manager.dslh.loc[dslh_ss_indices]

                output_file = _get_tmp_filename(settings, '3rd_stage_diverse_samples_from_phase1')
                excel_file = output_file.replace('.csv', '.xlsx')
                dslh_ss.to_excel(excel_file, index=False)

                _buffer.sort(key=lambda x: x.get('Hardness_Score', 0), reverse=True)
                _buffer = _buffer[:settings.phase_manager.SPE4]
                df = pd.DataFrame(_buffer)

                output_file = _get_tmp_filename(settings, '3rd_stage_top_hardness_samples')
                excel_file = output_file.replace('.csv', '.xlsx')
                df.to_excel(excel_file, index=False)

                df=pd.concat ([df, dslh_ss], ignore_index=True)
                settings.phase_manager.ds_phase4 = df
                output_file = _get_tmp_filename(settings, 'source_phase4')
                excel_file = output_file.replace('.csv', '.xlsx')
                df.to_excel(excel_file, index=False)

def _safe_str_list(value):
    """Safely convert lists or other types to string."""
    if isinstance(value, list):
        return ", ".join(map(str, value))
    elif value is None:
        return ""
    else:
        return str(value)

def calculate_hardness_scores(settings,_loss_matrix, _iou_matrix, alpha=0.7, beta=0.3):
        # if settings.phase_manager.number == 1 or settings.phase_manager.number == 3:
        avg_losses = np.mean(np.array(_loss_matrix).squeeze(), axis=0)
        Lmin = np.percentile(avg_losses, 1)
        Lmax = np.percentile(avg_losses, 99)
        avg_losses_tensor = torch.from_numpy(avg_losses).float()
        # Now use torch.clamp
        clipped_loss = torch.clamp(avg_losses_tensor, min=Lmin, max=Lmax).numpy()
        loss_norm = (clipped_loss - Lmin) / (Lmax - Lmin)

        ious_3d = np.array(_iou_matrix)
        avg_ious = np.mean(ious_3d[:, :, 0], axis=0)  # Shape: (samples,)
        Imin = np.percentile(avg_ious, 1)
        Imax = np.percentile(avg_ious, 99)
        avg_ious_tensor = torch.from_numpy(avg_ious).float()
        clipped_ious = torch.clamp(avg_ious_tensor, min=Imin, max=Imax).numpy()
        ious_norm = (clipped_ious - Imin) / (Imax - Imin)
        hardness_scores = alpha * loss_norm + beta * (1 - ious_norm)

        return hardness_scores
