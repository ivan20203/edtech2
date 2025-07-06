#!/usr/bin/env python3
"""
clean_dia_outliers.py
====================
Analyzes DIA DAC code files to find and remove outliers.

This script:
1. Loads all .dac.npy files
2. Analyzes corresponding sentences from sentences.txt
3. Calculates statistics (mean, std, min, max)
4. Identifies outliers using IQR method for frame counts and frame-to-word ratios
5. Removes outlier files
6. Creates a clean dataset
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_dac_codes(data_dir="data/train", sentences_file="sentences.txt"):
    """
    Analyze all DAC code files and return statistics.
    
    Args:
        data_dir: Directory containing .dac.npy files
        sentences_file: Path to sentences.txt file
        
    Returns:
        Dictionary with analysis results
    """
    data_path = Path(data_dir)
    sentences_path = Path(sentences_file)
    dac_files = sorted(data_path.glob("*.dac.npy"))
    
    if not dac_files:
        print("No .dac.npy files found!")
        return None
    
    # Load sentences
    if sentences_path.exists():
        sentences = sentences_path.read_text(encoding="utf-8").splitlines()
        print(f"Loaded {len(sentences)} sentences from {sentences_file}")
    else:
        print(f"Warning: {sentences_file} not found. Text analysis will be limited.")
        sentences = []
    
    print(f"Found {len(dac_files)} DIA DAC code files")
    
    # Load all files and collect statistics
    frame_counts = []
    word_counts = []
    frame_to_word_ratios = []
    file_info = {}
    
    for idx, dac_file in enumerate(dac_files):
        try:
            dac_codes = np.load(dac_file)
            # DAC codes are typically (frames, 9) format
            if dac_codes.ndim == 2:
                frame_count = dac_codes.shape[0]  # Number of frames
            else:
                frame_count = len(dac_codes)  # Fallback for 1D arrays
            frame_counts.append(frame_count)
            
            # Get corresponding sentence
            word_count = 0
            sentence_text = ""
            if idx < len(sentences):
                sentence_text = sentences[idx].strip()
                word_count = len(sentence_text.split())
                word_counts.append(word_count)
                
                # Calculate frame-to-word ratio
                if word_count > 0:
                    ratio = frame_count / word_count
                    frame_to_word_ratios.append(ratio)
                else:
                    frame_to_word_ratios.append(0)
            else:
                word_counts.append(0)
                frame_to_word_ratios.append(0)
            
            file_info[dac_file.name] = {
                'path': dac_file,
                'frame_count': frame_count,
                'word_count': word_count,
                'frame_to_word_ratio': frame_to_word_ratios[-1] if frame_to_word_ratios else 0,
                'file_size': dac_file.stat().st_size,
                'sentence': sentence_text,
                'sentence_idx': idx,
                'dac_shape': dac_codes.shape
            }
            
        except Exception as e:
            print(f"Error loading {dac_file}: {e}")
            continue
    
    # Calculate statistics
    frame_counts = np.array(frame_counts)
    word_counts = np.array(word_counts)
    frame_to_word_ratios = np.array(frame_to_word_ratios)
    
    # Filter out zero ratios for statistics
    valid_ratios = frame_to_word_ratios[frame_to_word_ratios > 0]
    
    stats = {
        'total_files': len(frame_counts),
        'mean_frames': np.mean(frame_counts),
        'std_frames': np.std(frame_counts),
        'min_frames': np.min(frame_counts),
        'max_frames': np.max(frame_counts),
        'median_frames': np.median(frame_counts),
        'q1_frames': np.percentile(frame_counts, 25),
        'q3_frames': np.percentile(frame_counts, 75),
        'iqr_frames': np.percentile(frame_counts, 75) - np.percentile(frame_counts, 25),
        'mean_words': np.mean(word_counts),
        'std_words': np.std(word_counts),
        'min_words': np.min(word_counts),
        'max_words': np.max(word_counts),
        'mean_ratio': np.mean(valid_ratios) if len(valid_ratios) > 0 else 0,
        'std_ratio': np.std(valid_ratios) if len(valid_ratios) > 0 else 0,
        'min_ratio': np.min(valid_ratios) if len(valid_ratios) > 0 else 0,
        'max_ratio': np.max(valid_ratios) if len(valid_ratios) > 0 else 0,
        'median_ratio': np.median(valid_ratios) if len(valid_ratios) > 0 else 0,
        'q1_ratio': np.percentile(valid_ratios, 25) if len(valid_ratios) > 0 else 0,
        'q3_ratio': np.percentile(valid_ratios, 75) if len(valid_ratios) > 0 else 0,
        'iqr_ratio': np.percentile(valid_ratios, 75) - np.percentile(valid_ratios, 25) if len(valid_ratios) > 0 else 0,
        'file_info': file_info,
        'frame_counts': frame_counts,
        'word_counts': word_counts,
        'frame_to_word_ratios': frame_to_word_ratios
    }
    
    return stats

def identify_outliers(stats, method='iqr', multiplier=1.5):
    """
    Identify outliers using specified method for both frame counts and frame-to-word ratios.
    
    Args:
        stats: Statistics dictionary from analyze_dac_codes
        method: 'iqr' or 'zscore'
        multiplier: Multiplier for IQR method (default 1.5)
        
    Returns:
        Dictionary with outlier information
    """
    file_info = stats['file_info']
    
    outliers = {
        'frame_count_outliers': [],
        'ratio_outliers': [],
        'combined_outliers': []
    }
    
    if method == 'iqr':
        # IQR method for frame counts
        q1_frames = stats['q1_frames']
        q3_frames = stats['q3_frames']
        iqr_frames = stats['iqr_frames']
        
        lower_bound_frames = q1_frames - multiplier * iqr_frames
        upper_bound_frames = q3_frames + multiplier * iqr_frames
        
        print(f"IQR Method for Frame Counts:")
        print(f"  Q1: {q1_frames:.1f}")
        print(f"  Q3: {q3_frames:.1f}")
        print(f"  IQR: {iqr_frames:.1f}")
        print(f"  Lower bound: {lower_bound_frames:.1f}")
        print(f"  Upper bound: {upper_bound_frames:.1f}")
        
        # IQR method for frame-to-word ratios
        if stats['iqr_ratio'] > 0:
            q1_ratio = stats['q1_ratio']
            q3_ratio = stats['q3_ratio']
            iqr_ratio = stats['iqr_ratio']
            
            lower_bound_ratio = q1_ratio - multiplier * iqr_ratio
            upper_bound_ratio = q3_ratio + multiplier * iqr_ratio
            
            print(f"\nIQR Method for Frame-to-Word Ratios:")
            print(f"  Q1: {q1_ratio:.3f}")
            print(f"  Q3: {q3_ratio:.3f}")
            print(f"  IQR: {iqr_ratio:.3f}")
            print(f"  Lower bound: {lower_bound_ratio:.3f}")
            print(f"  Upper bound: {upper_bound_ratio:.3f}")
        
        for filename, info in file_info.items():
            # Check frame count outliers
            if info['frame_count'] < lower_bound_frames or info['frame_count'] > upper_bound_frames:
                outliers['frame_count_outliers'].append(filename)
            
            # Check ratio outliers (only if ratio is valid)
            if info['frame_to_word_ratio'] > 0:
                if info['frame_to_word_ratio'] < lower_bound_ratio or info['frame_to_word_ratio'] > upper_bound_ratio:
                    outliers['ratio_outliers'].append(filename)
            
            # Combined outliers (either frame count or ratio is outlier)
            if (filename in outliers['frame_count_outliers'] or 
                filename in outliers['ratio_outliers']):
                outliers['combined_outliers'].append(filename)
                
    elif method == 'zscore':
        # Z-score method
        mean_frames = stats['mean_frames']
        std_frames = stats['std_frames']
        threshold = 2.0
        
        print(f"Z-score Method for Frame Counts:")
        print(f"  Mean: {mean_frames:.1f}")
        print(f"  Std: {std_frames:.1f}")
        print(f"  Threshold: ±{threshold}")
        
        if stats['std_ratio'] > 0:
            mean_ratio = stats['mean_ratio']
            std_ratio = stats['std_ratio']
            
            print(f"\nZ-score Method for Frame-to-Word Ratios:")
            print(f"  Mean: {mean_ratio:.3f}")
            print(f"  Std: {std_ratio:.3f}")
            print(f"  Threshold: ±{threshold}")
        
        for filename, info in file_info.items():
            # Check frame count outliers
            z_score_frames = abs((info['frame_count'] - mean_frames) / std_frames)
            if z_score_frames > threshold:
                outliers['frame_count_outliers'].append(filename)
            
            # Check ratio outliers
            if info['frame_to_word_ratio'] > 0 and stats['std_ratio'] > 0:
                z_score_ratio = abs((info['frame_to_word_ratio'] - mean_ratio) / std_ratio)
                if z_score_ratio > threshold:
                    outliers['ratio_outliers'].append(filename)
            
            # Combined outliers
            if (filename in outliers['frame_count_outliers'] or 
                filename in outliers['ratio_outliers']):
                outliers['combined_outliers'].append(filename)
    
    return outliers

def print_analysis(stats, outliers):
    """
    Print detailed analysis results.
    """
    print("\n" + "="*80)
    print("DIA DAC CODE ANALYSIS")
    print("="*80)
    
    print(f"Total files: {stats['total_files']}")
    print(f"Mean frames: {stats['mean_frames']:.1f}")
    print(f"Std frames: {stats['std_frames']:.1f}")
    print(f"Min frames: {stats['min_frames']}")
    print(f"Max frames: {stats['max_frames']}")
    print(f"Median frames: {stats['median_frames']:.1f}")
    
    if stats['mean_ratio'] > 0:
        print(f"\nFrame-to-Word Ratio Statistics:")
        print(f"Mean ratio: {stats['mean_ratio']:.3f}")
        print(f"Std ratio: {stats['std_ratio']:.3f}")
        print(f"Min ratio: {stats['min_ratio']:.3f}")
        print(f"Max ratio: {stats['max_ratio']:.3f}")
        print(f"Median ratio: {stats['median_ratio']:.3f}")
    
    print(f"\nOutliers found:")
    print(f"  Frame count outliers: {len(outliers['frame_count_outliers'])}")
    print(f"  Ratio outliers: {len(outliers['ratio_outliers'])}")
    print(f"  Combined outliers: {len(outliers['combined_outliers'])}")
    
    if outliers['combined_outliers']:
        print("\nOutlier files:")
        for outlier in sorted(outliers['combined_outliers']):
            info = stats['file_info'][outlier]
            reasons = []
            if outlier in outliers['frame_count_outliers']:
                reasons.append("frame_count")
            if outlier in outliers['ratio_outliers']:
                reasons.append("ratio")
            
            print(f"  {outlier}: {info['frame_count']} frames, {info['word_count']} words, "
                  f"ratio={info['frame_to_word_ratio']:.3f} ({', '.join(reasons)})")
            if info['sentence']:
                print(f"    Text: \"{info['sentence'][:60]}{'...' if len(info['sentence']) > 60 else ''}\"")
    
    # Show smallest files
    print(f"\nSmallest files (fewest frames):")
    sorted_files = sorted(stats['file_info'].items(), key=lambda x: x[1]['frame_count'])
    for i, (filename, info) in enumerate(sorted_files[:10]):  # Show top 10 smallest
        print(f"  {filename}: {info['frame_count']} frames, {info['word_count']} words, "
              f"ratio={info['frame_to_word_ratio']:.3f}")
        if info['sentence']:
            print(f"    Text: \"{info['sentence'][:60]}{'...' if len(info['sentence']) > 60 else ''}\"")
    
    # Show largest files
    print(f"\nLargest files (most frames):")
    sorted_files = sorted(stats['file_info'].items(), key=lambda x: x[1]['frame_count'], reverse=True)
    for i, (filename, info) in enumerate(sorted_files[:10]):  # Show top 10 largest
        print(f"  {filename}: {info['frame_count']} frames, {info['word_count']} words, "
              f"ratio={info['frame_to_word_ratio']:.3f}")
        if info['sentence']:
            print(f"    Text: \"{info['sentence'][:60]}{'...' if len(info['sentence']) > 60 else ''}\"")
    
    # Show distribution
    print(f"\nFrame count distribution:")
    counts = stats['frame_counts']
    print(f"  < 100 frames: {np.sum(counts < 100)} files")
    print(f"  100-200 frames: {np.sum((counts >= 100) & (counts < 200))} files")
    print(f"  200-300 frames: {np.sum((counts >= 200) & (counts < 300))} files")
    print(f"  300-400 frames: {np.sum((counts >= 300) & (counts < 400))} files")
    print(f"  400-500 frames: {np.sum((counts >= 400) & (counts < 500))} files")
    print(f"  > 500 frames: {np.sum(counts >= 500)} files")
    
    if stats['mean_ratio'] > 0:
        print(f"\nFrame-to-word ratio distribution:")
        ratios = stats['frame_to_word_ratios']
        valid_ratios = ratios[ratios > 0]
        print(f"  < 10 ratio: {np.sum(valid_ratios < 10)} files")
        print(f"  10-20 ratio: {np.sum((valid_ratios >= 10) & (valid_ratios < 20))} files")
        print(f"  20-30 ratio: {np.sum((valid_ratios >= 20) & (valid_ratios < 30))} files")
        print(f"  30-50 ratio: {np.sum((valid_ratios >= 30) & (valid_ratios < 50))} files")
        print(f"  > 50 ratio: {np.sum(valid_ratios >= 50)} files")

def remove_outliers(outliers, data_dir="data/train"):
    """
    Remove outlier files from the dataset.
    
    Args:
        outliers: Dictionary with outlier information
        data_dir: Directory containing the files
        
    Returns:
        Number of files removed
    """
    files_to_remove = outliers['combined_outliers']
    
    if not files_to_remove:
        print("No outliers to remove.")
        return 0
    
    data_path = Path(data_dir)
    removed_count = 0
    
    for outlier in files_to_remove:
        file_path = data_path / outlier
        
        if file_path.exists():
            # Delete file directly
            file_path.unlink()
            print(f"  Deleted {outlier}")
            removed_count += 1
        else:
            print(f"  Warning: {outlier} not found")
    
    return removed_count

def create_clean_dataset(data_dir="data/train", output_dir="data/train_clean", outliers=None):
    """
    Create a clean dataset by copying non-outlier files.
    
    Args:
        data_dir: Source directory
        output_dir: Output directory for clean dataset
        outliers: Dictionary with outlier information
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all .dac.npy files
    dac_files = list(data_path.glob("*.dac.npy"))
    
    if not dac_files:
        print("No .dac.npy files found!")
        return
    
    # Filter out outliers if provided
    if outliers:
        outlier_files = set(outliers['combined_outliers'])
        dac_files = [f for f in dac_files if f.name not in outlier_files]
        print(f"Filtering out {len(outlier_files)} outlier files")
    
    print(f"Creating clean dataset in {output_dir}")
    print(f"Copying {len(dac_files)} files...")
    
    for dac_file in dac_files:
        output_file = output_path / dac_file.name
        # Copy file
        import shutil
        shutil.copy2(dac_file, output_file)
    
    print(f"✓ Clean dataset created with {len(dac_files)} files")

def main():
    """
    Main function to run the outlier analysis and cleaning.
    """
    print("DIA DAC Code Outlier Analysis")
    print("="*50)
    
    # Analyze the data
    stats = analyze_dac_codes()
    if not stats:
        return
    
    # Identify outliers using IQR method
    outliers_iqr = identify_outliers(stats, method='iqr', multiplier=1.5)
    
    # Also try z-score method
    outliers_zscore = identify_outliers(stats, method='zscore')
    
    print(f"\nOutliers (IQR method): {len(outliers_iqr['combined_outliers'])}")
    print(f"Outliers (Z-score method): {len(outliers_zscore['combined_outliers'])}")
    
    # Use the more conservative method (fewer outliers)
    if len(outliers_iqr['combined_outliers']) <= len(outliers_zscore['combined_outliers']):
        outliers = outliers_iqr
        method_used = "IQR"
    else:
        outliers = outliers_zscore
        method_used = "Z-score"
    
    # Print analysis
    print_analysis(stats, outliers)
    
    # Ask user what to do
    print(f"\nOptions:")
    print("1. Remove outliers (permanent deletion)")
    print("2. Create clean dataset (copy non-outliers)")
    print("3. Just show analysis (no changes)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        confirm = input("Remove outliers? This will permanently delete files. (y/N): ").strip().lower()
        if confirm == 'y':
            removed = remove_outliers(outliers)
            print(f"✓ Removed {removed} outlier files")
        else:
            print("Operation cancelled.")
    
    elif choice == "2":
        output_dir = input("Enter output directory name (default: data/train_clean): ").strip()
        if not output_dir:
            output_dir = "data/train_clean"
        create_clean_dataset(output_dir=output_dir, outliers=outliers)
    
    elif choice == "3":
        print("Analysis complete. No files modified.")
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 