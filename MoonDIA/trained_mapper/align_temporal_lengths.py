#!/usr/bin/env python3
"""
align_temporal_lengths.py
=========================
Ensures temporal alignment between MoonCast semantic tokens and DIA DAC codes.

This script:
1. Loads all training pairs
2. Checks temporal alignment (same length after trimming)
3. Trims or pads sequences to match
4. Creates aligned dataset
5. Reports alignment statistics
"""

import numpy as np
import torch
from pathlib import Path
import shutil
import json
from typing import List, Tuple, Dict

def load_pair_info(data_dir: str) -> List[Dict]:
    """
    Load information about all training pairs.
    """
    data_path = Path(data_dir)
    
    # Find all pairs
    mc_files = sorted(data_path.glob("*.mc.npy"))
    dac_files = sorted(data_path.glob("*.dac.npy"))
    
    pairs = []
    for mc_file, dac_file in zip(mc_files, dac_files):
        try:
            # Load data
            mc_tokens = np.load(mc_file)
            dac_codes = np.load(dac_file)
            
            # Get file info - handle the .mc and .dac extensions properly
            mc_stem = mc_file.stem.replace('.mc', '')
            dac_stem = dac_file.stem.replace('.dac', '')
            
            if mc_stem != dac_stem:
                print(f"⚠️  Warning: Mismatched file names: {mc_file.name} vs {dac_file.name}")
                continue
            
            pairs.append({
                'stem': mc_stem,
                'mc_file': mc_file,
                'dac_file': dac_file,
                'mc_tokens': mc_tokens,
                'dac_codes': dac_codes,
                'mc_length': len(mc_tokens),
                'dac_length': dac_codes.shape[0],
                'dac_channels': dac_codes.shape[1] if len(dac_codes.shape) > 1 else 1
            })
            
        except Exception as e:
            print(f"Error loading {mc_file.name} or {dac_file.name}: {e}")
    
    return pairs

def analyze_temporal_alignment(pairs: List[Dict]) -> Dict:
    """
    Analyze temporal alignment between semantic tokens and DAC codes.
    """
    print("="*60)
    print("TEMPORAL ALIGNMENT ANALYSIS")
    print("="*60)
    
    # Calculate length ratios
    length_ratios = []
    alignment_issues = []
    well_aligned = []
    
    for pair in pairs:
        mc_len = pair['mc_length']
        dac_len = pair['dac_length']
        ratio = dac_len / mc_len
        
        length_ratios.append(ratio)
        
        # Check for alignment issues
        if abs(ratio - 1.0) > 0.1:  # More than 10% difference
            alignment_issues.append({
                'stem': pair['stem'],
                'mc_length': mc_len,
                'dac_length': dac_len,
                'ratio': ratio,
                'issue': f"Length mismatch: {mc_len} vs {dac_len} (ratio: {ratio:.3f})"
            })
        else:
            well_aligned.append(pair['stem'])
    
    # Statistics
    mean_ratio = np.mean(length_ratios)
    std_ratio = np.std(length_ratios)
    min_ratio = min(length_ratios)
    max_ratio = max(length_ratios)
    
    print(f"Total pairs: {len(pairs)}")
    print(f"Well-aligned pairs (within 10%): {len(well_aligned)}")
    print(f"Alignment issues: {len(alignment_issues)}")
    print(f"\nLength ratio statistics:")
    print(f"  Mean: {mean_ratio:.3f}")
    print(f"  Std: {std_ratio:.3f}")
    print(f"  Min: {min_ratio:.3f}")
    print(f"  Max: {max_ratio:.3f}")
    
    if alignment_issues:
        print(f"\n⚠️  ALIGNMENT ISSUES:")
        for issue in alignment_issues[:10]:  # Show first 10
            print(f"  {issue['stem']}: {issue['issue']}")
        if len(alignment_issues) > 10:
            print(f"  ... and {len(alignment_issues) - 10} more")
    
    return {
        'pairs': pairs,
        'length_ratios': length_ratios,
        'alignment_issues': alignment_issues,
        'well_aligned': well_aligned,
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio
    }

def stretch(arr, target_len):
    """
    Stretch array to target length using linear interpolation.
    """
    if len(arr) == target_len:
        return arr
    idx = np.linspace(0, len(arr)-1, target_len).astype(int)
    return arr[idx]

def trim_to_minimum_length(pairs: List[Dict]) -> List[Dict]:
    """
    Align pairs by stretching MC tokens to DAC length (Strategy B).
    This keeps all voiced content from DIA while maintaining 1:1 alignment.
    """
    print("\n" + "="*60)
    print("ALIGNING WITH STRETCH STRATEGY")
    print("="*60)
    
    trimmed_pairs = []
    stretch_count = 0
    trim_count = 0
    perfect_count = 0
    
    for pair in pairs:
        mc_len = pair['mc_length']
        dac_len = pair['dac_length']
        
        if mc_len == dac_len:
            # Perfect alignment - no changes needed
            trimmed_mc = pair['mc_tokens']
            trimmed_dac = pair['dac_codes']
            strategy = 'perfect'
            perfect_count += 1
            
        elif mc_len < dac_len:
            # Stretch MC tokens to match DAC length (keep all voiced content)
            trimmed_mc = stretch(pair['mc_tokens'], dac_len)
            trimmed_dac = pair['dac_codes']
            strategy = 'stretch'
            stretch_count += 1
            
        else:
            # Rare case: MC longer than DAC - trim MC to DAC length
            trimmed_mc = pair['mc_tokens'][:dac_len]
            trimmed_dac = pair['dac_codes']
            strategy = 'trim'
            trim_count += 1
        
        trimmed_pairs.append({
            'stem': pair['stem'],
            'mc_file': pair['mc_file'],
            'dac_file': pair['dac_file'],
            'mc_tokens': trimmed_mc,
            'dac_codes': trimmed_dac,
            'original_mc_length': mc_len,
            'original_dac_length': dac_len,
            'trimmed_length': len(trimmed_mc),
            'trimmed_mc_length': len(trimmed_mc),
            'trimmed_dac_length': trimmed_dac.shape[0],
            'strategy': strategy
        })
    
    print(f"Aligned {len(trimmed_pairs)} pairs")
    print(f"  - Perfect alignment: {perfect_count}")
    print(f"  - Stretched MC tokens: {stretch_count}")
    print(f"  - Trimmed MC tokens: {trim_count}")
    
    return trimmed_pairs

def create_aligned_dataset(trimmed_pairs: List[Dict], output_dir: str = "data/train_aligned") -> str:
    """
    Create aligned dataset with trimmed pairs.
    """
    print("\n" + "="*60)
    print("CREATING ALIGNED DATASET")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Verify alignment
    alignment_verified = []
    alignment_failed = []
    
    for pair in trimmed_pairs:
        mc_len = pair['trimmed_mc_length']
        dac_len = pair['trimmed_dac_length']
        
        if mc_len == dac_len:
            alignment_verified.append(pair)
            
            # Save aligned files
            mc_output = output_path / f"{pair['stem']}.mc.npy"
            dac_output = output_path / f"{pair['stem']}.dac.npy"
            
            np.save(mc_output, pair['mc_tokens'])
            np.save(dac_output, pair['dac_codes'])
            
        else:
            alignment_failed.append(pair)
    
    print(f"Alignment verified: {len(alignment_verified)} pairs")
    print(f"Alignment failed: {len(alignment_failed)} pairs")
    
    if alignment_failed:
        print(f"\n❌ ALIGNMENT FAILURES:")
        for pair in alignment_failed:
            print(f"  {pair['stem']}: MC={pair['trimmed_mc_length']}, DAC={pair['trimmed_dac_length']}")
    
    return str(output_path)

def analyze_aligned_dataset(aligned_pairs: List[Dict]) -> Dict:
    """
    Analyze the final aligned dataset.
    """
    print("\n" + "="*60)
    print("ALIGNED DATASET ANALYSIS")
    print("="*60)
    
    lengths = [p['trimmed_length'] for p in aligned_pairs]
    mc_lengths = [p['trimmed_mc_length'] for p in aligned_pairs]
    dac_lengths = [p['trimmed_dac_length'] for p in aligned_pairs]
    
    print(f"Total aligned pairs: {len(aligned_pairs)}")
    print(f"Length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median: {np.median(lengths):.2f}")
    print(f"  Std: {np.std(lengths):.2f}")
    
    # Check for perfect alignment
    perfect_alignment = all(mc == dac for mc, dac in zip(mc_lengths, dac_lengths))
    print(f"\nPerfect alignment: {'✅ YES' if perfect_alignment else '❌ NO'}")
    
    # Vocabulary analysis
    all_mc_tokens = np.concatenate([p['mc_tokens'] for p in aligned_pairs])
    all_dac_codes = np.concatenate([p['dac_codes'] for p in aligned_pairs])
    
    mc_vocab_size = len(np.unique(all_mc_tokens))
    dac_vocab_size = len(np.unique(all_dac_codes))
    
    print(f"\nVocabulary sizes:")
    print(f"  Semantic tokens: {mc_vocab_size}")
    print(f"  DAC codes: {dac_vocab_size}")
    
    return {
        'aligned_pairs': aligned_pairs,
        'lengths': lengths,
        'perfect_alignment': perfect_alignment,
        'mc_vocab_size': mc_vocab_size,
        'dac_vocab_size': dac_vocab_size
    }

def save_alignment_report(analysis: Dict, alignment_analysis: Dict, aligned_analysis: Dict, output_dir: str):
    """
    Save detailed alignment report.
    """
    report = {
        'original_analysis': {
            'total_pairs': len(analysis['pairs']),
            'well_aligned': len(analysis['well_aligned']),
            'alignment_issues': len(analysis['alignment_issues']),
            'mean_ratio': float(analysis['mean_ratio']),
            'std_ratio': float(analysis['std_ratio'])
        },
        'trimming_results': {
            'total_trimmed_pairs': len(alignment_analysis['pairs']),
            'alignment_verified': len([p for p in alignment_analysis['pairs'] 
                                     if p['trimmed_mc_length'] == p['trimmed_dac_length']]),
            'alignment_failed': len([p for p in alignment_analysis['pairs'] 
                                   if p['trimmed_mc_length'] != p['trimmed_dac_length']])
        },
        'final_dataset': {
            'aligned_pairs': len(aligned_analysis['aligned_pairs']),
            'perfect_alignment': aligned_analysis['perfect_alignment'],
            'length_stats': {
                'min': min(aligned_analysis['lengths']),
                'max': max(aligned_analysis['lengths']),
                'mean': float(np.mean(aligned_analysis['lengths'])),
                'median': float(np.median(aligned_analysis['lengths'])),
                'std': float(np.std(aligned_analysis['lengths']))
            },
            'vocab_sizes': {
                'semantic': aligned_analysis['mc_vocab_size'],
                'dac': aligned_analysis['dac_vocab_size']
            }
        },
        'alignment_issues': [
            {
                'stem': issue['stem'],
                'mc_length': issue['mc_length'],
                'dac_length': issue['dac_length'],
                'ratio': float(issue['ratio']),
                'issue': issue['issue']
            }
            for issue in analysis['alignment_issues']
        ]
    }
    
    report_path = Path(output_dir) / "alignment_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Alignment report saved to {report_path}")

def main():
    """
    Main function to align temporal lengths.
    """
    print("Temporal Length Alignment for Semantic-to-DAC Mapper")
    print("="*60)
    
    # Load and analyze original data
    pairs = load_pair_info("data/train")
    if not pairs:
        print("❌ No training pairs found!")
        return
    
    # Analyze temporal alignment
    analysis = analyze_temporal_alignment(pairs)
    
    # Ask user what to do
    print(f"\nOptions:")
    print("1. Trim to minimum length (recommended)")
    print("2. Skip pairs with alignment issues")
    print("3. Just show analysis (no changes)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Trim to minimum length
        trimmed_pairs = trim_to_minimum_length(pairs)
        
        # Create aligned dataset
        output_dir = create_aligned_dataset(trimmed_pairs)
        
        # Analyze final dataset
        aligned_analysis = analyze_aligned_dataset(trimmed_pairs)
        
        # Save report
        save_alignment_report(analysis, {'pairs': trimmed_pairs}, aligned_analysis, output_dir)
        
        print(f"\n✅ Aligned dataset created in: {output_dir}")
        print(f"Ready for training with {len(aligned_analysis['aligned_pairs'])} pairs")
        
    elif choice == "2":
        # Skip problematic pairs
        good_pairs = [p for p in pairs if p['stem'] in analysis['well_aligned']]
        print(f"\nKeeping {len(good_pairs)} well-aligned pairs")
        
        # Create dataset with only well-aligned pairs
        output_dir = create_aligned_dataset(good_pairs, "data/train_well_aligned")
        
        # Analyze final dataset
        aligned_analysis = analyze_aligned_dataset(good_pairs)
        
        # Save report
        save_alignment_report(analysis, {'pairs': good_pairs}, aligned_analysis, output_dir)
        
        print(f"\n✅ Well-aligned dataset created in: {output_dir}")
        print(f"Ready for training with {len(aligned_analysis['aligned_pairs'])} pairs")
        
    elif choice == "3":
        print("Analysis complete. No changes made.")
        
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 