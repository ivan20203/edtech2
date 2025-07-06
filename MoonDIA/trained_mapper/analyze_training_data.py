#!/usr/bin/env python3
"""
analyze_training_data.py
========================
Analyzes the training data for the semantic-to-DAC mapper to identify potential issues
and provide insights for training the neural network.
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

def load_and_analyze_data(data_dir="data/train"):
    """
    Load and analyze the training data comprehensively.
    """
    data_path = Path(data_dir)
    
    # Find all pairs
    mc_files = sorted(data_path.glob("*.mc.npy"))
    dac_files = sorted(data_path.glob("*.dac.npy"))
    
    print(f"Found {len(mc_files)} MoonCast files and {len(dac_files)} DAC files")
    
    if len(mc_files) != len(dac_files):
        print("⚠️  WARNING: Mismatch in file counts!")
        return None
    
    # Load all data
    semantic_tokens = []
    dac_codes = []
    file_info = []
    
    for i, (mc_file, dac_file) in enumerate(zip(mc_files, dac_files)):
        try:
            # Load semantic tokens
            sem = np.load(mc_file)
            semantic_tokens.append(sem)
            
            # Load DAC codes
            dac = np.load(dac_file)
            dac_codes.append(dac)
            
            # Store file info
            file_info.append({
                'index': i,
                'mc_file': mc_file.name,
                'dac_file': dac_file.name,
                'mc_length': len(sem),
                'dac_length': dac.shape[0],
                'dac_channels': dac.shape[1] if len(dac.shape) > 1 else 1,
                'mc_unique_tokens': len(np.unique(sem)),
                'dac_unique_codes': len(np.unique(dac)),
                'mc_min': int(sem.min()),
                'mc_max': int(sem.max()),
                'dac_min': int(dac.min()),
                'dac_max': int(dac.max()),
                'mc_file_size': mc_file.stat().st_size,
                'dac_file_size': dac_file.stat().st_size
            })
            
        except Exception as e:
            print(f"Error loading {mc_file.name} or {dac_file.name}: {e}")
    
    return semantic_tokens, dac_codes, file_info

def analyze_semantic_tokens(semantic_tokens, file_info):
    """
    Analyze MoonCast semantic tokens.
    """
    print("\n" + "="*60)
    print("MOONCAST SEMANTIC TOKENS ANALYSIS")
    print("="*60)
    
    # Basic statistics
    lengths = [len(sem) for sem in semantic_tokens]
    unique_counts = [info['mc_unique_tokens'] for info in file_info]
    min_vals = [info['mc_min'] for info in file_info]
    max_vals = [info['mc_max'] for info in file_info]
    
    print(f"Total samples: {len(semantic_tokens)}")
    print(f"Length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median: {np.median(lengths):.2f}")
    print(f"  Std: {np.std(lengths):.2f}")
    
    print(f"\nToken value statistics:")
    print(f"  Global min: {min(min_vals)}")
    print(f"  Global max: {max(max_vals)}")
    print(f"  Mean unique tokens per sample: {np.mean(unique_counts):.2f}")
    
    # Check for vocabulary size
    all_tokens = np.concatenate(semantic_tokens)
    vocab_size = len(np.unique(all_tokens))
    print(f"  Total vocabulary size: {vocab_size}")
    
    # Check for potential issues
    issues = []
    
    if vocab_size > 16384:
        issues.append(f"Vocabulary size ({vocab_size}) exceeds model capacity (16384)")
    
    if min(min_vals) < 0:
        issues.append("Negative token values found")
    
    if max(max_vals) >= 16384:
        issues.append(f"Token values exceed vocabulary size (max: {max(max_vals)})")
    
    # Check for extreme length variations
    length_std = np.std(lengths)
    length_mean = np.mean(lengths)
    if length_std / length_mean > 0.5:
        issues.append("High length variation may cause training issues")
    
    if issues:
        print(f"\n⚠️  POTENTIAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✅ No obvious issues detected")
    
    return {
        'lengths': lengths,
        'vocab_size': vocab_size,
        'all_tokens': all_tokens,
        'issues': issues
    }

def analyze_dac_codes(dac_codes, file_info):
    """
    Analyze DIA DAC codes.
    """
    print("\n" + "="*60)
    print("DIA DAC CODES ANALYSIS")
    print("="*60)
    
    # Basic statistics
    lengths = [dac.shape[0] for dac in dac_codes]
    channels = [dac.shape[1] for dac in dac_codes]
    unique_counts = [info['dac_unique_codes'] for info in file_info]
    min_vals = [info['dac_min'] for info in file_info]
    max_vals = [info['dac_max'] for info in file_info]
    
    print(f"Total samples: {len(dac_codes)}")
    print(f"Length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median: {np.median(lengths):.2f}")
    print(f"  Std: {np.std(lengths):.2f}")
    
    print(f"\nChannel statistics:")
    print(f"  Expected channels: 9")
    print(f"  Actual channels: {set(channels)}")
    
    print(f"\nCode value statistics:")
    print(f"  Global min: {min(min_vals)}")
    print(f"  Global max: {max(max_vals)}")
    print(f"  Mean unique codes per sample: {np.mean(unique_counts):.2f}")
    
    # Check for vocabulary size
    all_codes = np.concatenate(dac_codes)
    vocab_size = len(np.unique(all_codes))
    print(f"  Total vocabulary size: {vocab_size}")
    
    # Check for potential issues
    issues = []
    
    if set(channels) != {9}:
        issues.append(f"Unexpected channel count: {set(channels)}")
    
    if vocab_size > 1024:
        issues.append(f"Vocabulary size ({vocab_size}) exceeds typical DAC capacity (1024)")
    
    if min(min_vals) < 0:
        issues.append("Negative code values found")
    
    if max(max_vals) >= 1024:
        issues.append(f"Code values exceed vocabulary size (max: {max(max_vals)})")
    
    # Check for extreme length variations
    length_std = np.std(lengths)
    length_mean = np.mean(lengths)
    if length_std / length_mean > 0.5:
        issues.append("High length variation may cause training issues")
    
    if issues:
        print(f"\n⚠️  POTENTIAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✅ No obvious issues detected")
    
    return {
        'lengths': lengths,
        'channels': channels,
        'vocab_size': vocab_size,
        'all_codes': all_codes,
        'issues': issues
    }

def analyze_length_correlations(semantic_tokens, dac_codes, file_info):
    """
    Analyze correlations between semantic token and DAC code lengths.
    """
    print("\n" + "="*60)
    print("LENGTH CORRELATION ANALYSIS")
    print("="*60)
    
    mc_lengths = [len(sem) for sem in semantic_tokens]
    dac_lengths = [dac.shape[0] for dac in dac_codes]
    
    # Calculate correlation
    correlation = np.corrcoef(mc_lengths, dac_lengths)[0, 1]
    print(f"Length correlation: {correlation:.4f}")
    
    if correlation > 0.7:
        print("✅ Strong positive correlation - good for training")
    elif correlation > 0.3:
        print("⚠️  Moderate correlation - training may be challenging")
    else:
        print("❌ Weak correlation - training will be difficult")
    
    # Check for length mismatches
    length_ratios = [dac_len / mc_len for mc_len, dac_len in zip(mc_lengths, dac_lengths)]
    mean_ratio = np.mean(length_ratios)
    std_ratio = np.std(length_ratios)
    
    print(f"\nLength ratios (DAC/MC):")
    print(f"  Mean: {mean_ratio:.4f}")
    print(f"  Std: {std_ratio:.4f}")
    print(f"  Min: {min(length_ratios):.4f}")
    print(f"  Max: {max(length_ratios):.4f}")
    
    # Find outliers
    outlier_threshold = 2.0
    outliers = [(i, ratio) for i, ratio in enumerate(length_ratios) 
                if abs(ratio - mean_ratio) > outlier_threshold * std_ratio]
    
    if outliers:
        print(f"\n⚠️  Length ratio outliers (threshold: {outlier_threshold}σ):")
        for idx, ratio in outliers[:10]:  # Show first 10
            print(f"  Sample {idx}: {ratio:.4f} ({file_info[idx]['mc_file']})")
        if len(outliers) > 10:
            print(f"  ... and {len(outliers) - 10} more")
    
    return {
        'correlation': correlation,
        'length_ratios': length_ratios,
        'outliers': outliers
    }

def analyze_training_viability(semantic_analysis, dac_analysis, correlation_analysis):
    """
    Assess overall training viability.
    """
    print("\n" + "="*60)
    print("TRAINING VIABILITY ASSESSMENT")
    print("="*60)
    
    issues = []
    warnings = []
    recommendations = []
    
    # Check data size
    if len(semantic_analysis['lengths']) < 50:
        issues.append("Very small dataset (< 50 samples)")
        recommendations.append("Consider collecting more training data")
    elif len(semantic_analysis['lengths']) < 100:
        warnings.append("Small dataset (< 100 samples)")
        recommendations.append("Consider data augmentation or transfer learning")
    
    # Check vocabulary sizes
    if semantic_analysis['vocab_size'] > 16384:
        issues.append("Semantic vocabulary too large")
        recommendations.append("Consider vocabulary reduction or larger model")
    
    if dac_analysis['vocab_size'] > 1024:
        issues.append("DAC vocabulary too large")
        recommendations.append("Check DAC code generation parameters")
    
    # Check length correlation
    if correlation_analysis['correlation'] < 0.3:
        issues.append("Weak length correlation")
        recommendations.append("Consider different model architecture or data preprocessing")
    
    # Check length variations
    mc_cv = np.std(semantic_analysis['lengths']) / np.mean(semantic_analysis['lengths'])
    dac_cv = np.std(dac_analysis['lengths']) / np.mean(dac_analysis['lengths'])
    
    if mc_cv > 0.5:
        warnings.append("High semantic token length variation")
        recommendations.append("Consider length-based batching or padding strategies")
    
    if dac_cv > 0.5:
        warnings.append("High DAC code length variation")
        recommendations.append("Consider length-based batching or padding strategies")
    
    # Overall assessment
    if issues:
        print("❌ CRITICAL ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    # Training confidence
    if not issues and not warnings:
        print("\n✅ TRAINING READY - No major issues detected")
        confidence = "High"
    elif not issues:
        print("\n⚠️  TRAINING POSSIBLE - Some warnings, but no critical issues")
        confidence = "Medium"
    else:
        print("\n❌ TRAINING NOT RECOMMENDED - Critical issues must be resolved")
        confidence = "Low"
    
    return {
        'confidence': confidence,
        'issues': issues,
        'warnings': warnings,
        'recommendations': recommendations
    }

def create_visualizations(semantic_analysis, dac_analysis, correlation_analysis, file_info):
    """
    Create visualizations of the data.
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    try:
        # Set up the plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Data Analysis', fontsize=16)
        
        # 1. Semantic token length distribution
        axes[0, 0].hist(semantic_analysis['lengths'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Semantic Token Lengths')
        axes[0, 0].set_xlabel('Length')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. DAC code length distribution
        axes[0, 1].hist(dac_analysis['lengths'], bins=20, alpha=0.7, color='red')
        axes[0, 1].set_title('DAC Code Lengths')
        axes[0, 1].set_xlabel('Length')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Length correlation scatter plot
        mc_lengths = semantic_analysis['lengths']
        dac_lengths = dac_analysis['lengths']
        axes[0, 2].scatter(mc_lengths, dac_lengths, alpha=0.6)
        axes[0, 2].set_title(f'Length Correlation: {correlation_analysis["correlation"]:.3f}')
        axes[0, 2].set_xlabel('Semantic Length')
        axes[0, 2].set_ylabel('DAC Length')
        
        # 4. Length ratio distribution
        axes[1, 0].hist(correlation_analysis['length_ratios'], bins=20, alpha=0.7, color='green')
        axes[1, 0].set_title('Length Ratios (DAC/MC)')
        axes[1, 0].set_xlabel('Ratio')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Token value distributions
        axes[1, 1].hist(semantic_analysis['all_tokens'], bins=50, alpha=0.7, color='blue', label='Semantic')
        axes[1, 1].set_title('Token Value Distributions')
        axes[1, 1].set_xlabel('Token Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        # 6. Code value distributions
        axes[1, 2].hist(dac_analysis['all_codes'], bins=50, alpha=0.7, color='red', label='DAC')
        axes[1, 2].set_title('Code Value Distributions')
        axes[1, 2].set_xlabel('Code Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('training_data_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizations saved to 'training_data_analysis.png'")
        
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")

def save_analysis_report(semantic_analysis, dac_analysis, correlation_analysis, viability, file_info):
    """
    Save detailed analysis report to JSON.
    """
    report = {
        'summary': {
            'total_samples': len(file_info),
            'semantic_vocab_size': semantic_analysis['vocab_size'],
            'dac_vocab_size': dac_analysis['vocab_size'],
            'length_correlation': correlation_analysis['correlation'],
            'training_confidence': viability['confidence']
        },
        'semantic_analysis': {
            'length_stats': {
                'min': min(semantic_analysis['lengths']),
                'max': max(semantic_analysis['lengths']),
                'mean': float(np.mean(semantic_analysis['lengths'])),
                'median': float(np.median(semantic_analysis['lengths'])),
                'std': float(np.std(semantic_analysis['lengths']))
            },
            'vocab_size': semantic_analysis['vocab_size'],
            'issues': semantic_analysis['issues']
        },
        'dac_analysis': {
            'length_stats': {
                'min': min(dac_analysis['lengths']),
                'max': max(dac_analysis['lengths']),
                'mean': float(np.mean(dac_analysis['lengths'])),
                'median': float(np.median(dac_analysis['lengths'])),
                'std': float(np.std(dac_analysis['lengths']))
            },
            'vocab_size': dac_analysis['vocab_size'],
            'channels': list(set(dac_analysis['channels'])),
            'issues': dac_analysis['issues']
        },
        'correlation_analysis': {
            'correlation': float(correlation_analysis['correlation']),
            'length_ratio_stats': {
                'mean': float(np.mean(correlation_analysis['length_ratios'])),
                'std': float(np.std(correlation_analysis['length_ratios'])),
                'min': float(min(correlation_analysis['length_ratios'])),
                'max': float(max(correlation_analysis['length_ratios']))
            },
            'outlier_count': len(correlation_analysis['outliers'])
        },
        'viability': {
            'confidence': viability['confidence'],
            'issues': viability['issues'],
            'warnings': viability['warnings'],
            'recommendations': viability['recommendations']
        },
        'file_details': file_info
    }
    
    with open('training_data_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Detailed report saved to 'training_data_report.json'")

def main():
    """
    Main analysis function.
    """
    print("Training Data Analysis for Semantic-to-DAC Mapper")
    print("="*60)
    
    # Load and analyze data
    result = load_and_analyze_data()
    if result is None:
        return
    
    semantic_tokens, dac_codes, file_info = result
    
    # Perform analyses
    semantic_analysis = analyze_semantic_tokens(semantic_tokens, file_info)
    dac_analysis = analyze_dac_codes(dac_codes, file_info)
    correlation_analysis = analyze_length_correlations(semantic_tokens, dac_codes, file_info)
    viability = analyze_training_viability(semantic_analysis, dac_analysis, correlation_analysis)
    
    # Create visualizations
    create_visualizations(semantic_analysis, dac_analysis, correlation_analysis, file_info)
    
    # Save detailed report
    save_analysis_report(semantic_analysis, dac_analysis, correlation_analysis, viability, file_info)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Training confidence: {viability['confidence']}")
    print(f"Files generated:")
    print(f"  - training_data_analysis.png (visualizations)")
    print(f"  - training_data_report.json (detailed report)")

if __name__ == "__main__":
    main() 