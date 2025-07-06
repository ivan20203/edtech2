#!/usr/bin/env python3
"""
clean_orphaned_pairs.py
======================
Finds and removes orphaned DAC files that don't have corresponding MoonCast semantic token files.

This script:
1. Scans data/train directory for .mc.npy and .dac.npy files
2. Identifies DAC files without corresponding MoonCast files
3. Removes orphaned DAC files
4. Reports statistics on the cleaning process
"""

import numpy as np
import os
from pathlib import Path
import shutil

def find_orphaned_files(data_dir="data/train"):
    """
    Find orphaned DAC files that don't have corresponding MoonCast files.
    
    Args:
        data_dir: Directory containing the training data
        
    Returns:
        Dictionary with analysis results
    """
    data_path = Path(data_dir)
    
    # Find all MoonCast and DAC files
    mc_files = set(f.stem.replace('.mc', '') for f in data_path.glob("*.mc.npy"))
    dac_files = set(f.stem.replace('.dac', '') for f in data_path.glob("*.dac.npy"))
    
    print(f"Found {len(mc_files)} MoonCast semantic token files")
    print(f"Found {len(dac_files)} DIA DAC code files")
    
    # Find orphaned DAC files (DAC files without corresponding MC files)
    orphaned_dac = dac_files - mc_files
    
    # Find orphaned MC files (MC files without corresponding DAC files)
    orphaned_mc = mc_files - dac_files
    
    # Find complete pairs
    complete_pairs = mc_files & dac_files
    
    return {
        'mc_files': mc_files,
        'dac_files': dac_files,
        'orphaned_dac': orphaned_dac,
        'orphaned_mc': orphaned_mc,
        'complete_pairs': complete_pairs,
        'total_mc': len(mc_files),
        'total_dac': len(dac_files),
        'orphaned_dac_count': len(orphaned_dac),
        'orphaned_mc_count': len(orphaned_mc),
        'complete_pairs_count': len(complete_pairs)
    }

def print_analysis(results):
    """
    Print detailed analysis of the file pairs.
    """
    print("\n" + "="*60)
    print("FILE PAIR ANALYSIS")
    print("="*60)
    
    print(f"Total MoonCast files: {results['total_mc']}")
    print(f"Total DAC files: {results['total_dac']}")
    print(f"Complete pairs: {results['complete_pairs_count']}")
    print(f"Orphaned DAC files: {results['orphaned_dac_count']}")
    print(f"Orphaned MC files: {results['orphaned_mc_count']}")
    
    if results['orphaned_dac']:
        print(f"\nOrphaned DAC files (will be deleted):")
        for orphan in sorted(results['orphaned_dac']):
            print(f"  {orphan}.dac.npy")
    
    if results['orphaned_mc']:
        print(f"\nOrphaned MC files (missing DAC partners):")
        for orphan in sorted(results['orphaned_mc']):
            print(f"  {orphan}.mc.npy")
    
    if results['complete_pairs']:
        print(f"\nComplete pairs (will be kept):")
        for pair in sorted(list(results['complete_pairs']))[:10]:  # Show first 10
            print(f"  {pair}.mc.npy ↔ {pair}.dac.npy")
        if len(results['complete_pairs']) > 10:
            print(f"  ... and {len(results['complete_pairs']) - 10} more pairs")

def remove_orphaned_dac_files(orphaned_dac, data_dir="data/train", backup=True):
    """
    Remove orphaned DAC files.
    
    Args:
        orphaned_dac: Set of orphaned DAC file stems
        data_dir: Directory containing the files
        backup: Whether to backup files before deletion
        
    Returns:
        Number of files removed
    """
    if not orphaned_dac:
        print("No orphaned DAC files to remove.")
        return 0
    
    data_path = Path(data_dir)
    removed_count = 0
    
    if backup:
        backup_dir = data_path / "backup_orphaned"
        backup_dir.mkdir(exist_ok=True)
        print(f"Backing up orphaned files to {backup_dir}")
    
    for orphan in orphaned_dac:
        dac_file = data_path / f"{orphan}.dac.npy"
        
        if dac_file.exists():
            if backup:
                # Move to backup directory
                backup_file = backup_dir / f"{orphan}.dac.npy"
                dac_file.rename(backup_file)
                print(f"  Moved {orphan}.dac.npy to backup")
            else:
                # Delete file directly
                dac_file.unlink()
                print(f"  Deleted {orphan}.dac.npy")
            
            removed_count += 1
        else:
            print(f"  Warning: {orphan}.dac.npy not found")
    
    return removed_count

def create_clean_dataset(data_dir="data/train", output_dir="data/train_clean"):
    """
    Create a clean dataset with only complete pairs.
    
    Args:
        data_dir: Source directory
        output_dir: Output directory for clean dataset
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find complete pairs
    results = find_orphaned_files(data_dir)
    complete_pairs = results['complete_pairs']
    
    if not complete_pairs:
        print("No complete pairs found!")
        return
    
    print(f"Creating clean dataset in {output_dir}")
    print(f"Copying {len(complete_pairs)} complete pairs...")
    
    copied_count = 0
    
    for pair in complete_pairs:
        mc_file = data_path / f"{pair}.mc.npy"
        dac_file = data_path / f"{pair}.dac.npy"
        
        if mc_file.exists() and dac_file.exists():
            # Copy both files
            shutil.copy2(mc_file, output_path / f"{pair}.mc.npy")
            shutil.copy2(dac_file, output_path / f"{pair}.dac.npy")
            copied_count += 1
        else:
            print(f"  Warning: Incomplete pair {pair}")
    
    print(f"✓ Clean dataset created with {copied_count} complete pairs")

def main():
    """
    Main function to run the orphaned file analysis and cleaning.
    """
    print("Orphaned File Pair Analysis")
    print("="*50)
    
    # Analyze the data
    results = find_orphaned_files()
    
    # Print analysis
    print_analysis(results)
    
    # Ask user what to do
    print(f"\nOptions:")
    print("1. Remove orphaned DAC files (backup first)")
    print("2. Remove orphaned DAC files (permanent deletion)")
    print("3. Create clean dataset (copy only complete pairs)")
    print("4. Just show analysis (no changes)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        if results['orphaned_dac_count'] > 0:
            confirm = input(f"Remove {results['orphaned_dac_count']} orphaned DAC files? This will backup files first. (y/N): ").strip().lower()
            if confirm == 'y':
                removed = remove_orphaned_dac_files(results['orphaned_dac'], backup=True)
                print(f"✓ Removed {removed} orphaned DAC files (backed up)")
            else:
                print("Operation cancelled.")
        else:
            print("No orphaned DAC files to remove.")
    
    elif choice == "2":
        if results['orphaned_dac_count'] > 0:
            confirm = input(f"Remove {results['orphaned_dac_count']} orphaned DAC files? This will permanently delete files. (y/N): ").strip().lower()
            if confirm == 'y':
                removed = remove_orphaned_dac_files(results['orphaned_dac'], backup=False)
                print(f"✓ Removed {removed} orphaned DAC files")
            else:
                print("Operation cancelled.")
        else:
            print("No orphaned DAC files to remove.")
    
    elif choice == "3":
        output_dir = input("Enter output directory name (default: data/train_clean): ").strip()
        if not output_dir:
            output_dir = "data/train_clean"
        create_clean_dataset(output_dir=output_dir)
    
    elif choice == "4":
        print("Analysis complete. No files modified.")
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 