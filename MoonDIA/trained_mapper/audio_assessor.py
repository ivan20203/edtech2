#!/usr/bin/env python3
"""
whisper.py
==========
Audio Quality Assessment using Whisper

This script:
1. Loads original sentences from sentences.txt
2. Transcribes DIA and MoonCast audio files using Whisper
3. Compares transcriptions with original text
4. Identifies satisfactory audio pairs (both DIA and MoonCast work well)
5. Outputs a list of working pairs
"""

import whisper
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
import argparse
import time
import difflib
import json
from typing import List, Dict, Tuple, Optional
import sys
import os

class AudioQualityAssessor:
    def __init__(self, model_size: str = "base"):
        """Initialize the audio quality assessor with Whisper model."""
        print(f"Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        print(f"✅ Whisper model loaded: {model_size}")
        
        # Quality thresholds
        self.min_similarity = 0.6  # Minimum text similarity score
        self.min_duration_ratio = 0.3  # Minimum audio duration relative to text length
        self.max_duration_ratio = 3.0  # Maximum audio duration relative to text length
        
    def load_sentences(self, sentences_file: str) -> Dict[int, str]:
        """Load sentences from file and create mapping by index."""
        sentences = {}
        try:
            with open(sentences_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        sentences[i] = line
            print(f"✅ Loaded {len(sentences)} sentences from {sentences_file}")
            return sentences
        except Exception as e:
            print(f"❌ Error loading sentences: {e}")
            return {}
    
    def transcribe_audio(self, audio_path: str) -> Optional[Dict]:
        """Transcribe audio file using Whisper."""
        try:
            # Get actual audio duration using soundfile
            audio_info = sf.info(audio_path)
            actual_duration = audio_info.duration
            
            result = self.model.transcribe(audio_path)
            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'en'),
                'segments': result.get('segments', []),
                'duration': actual_duration  # Use actual file duration instead of Whisper's
            }
        except Exception as e:
            print(f"  ❌ Transcription error: {e}")
            return None
    
    def calculate_similarity(self, original: str, transcription: str) -> float:
        """Calculate similarity between original text and transcription."""
        # Normalize both texts
        original_norm = original.lower().strip()
        transcription_norm = transcription.lower().strip()
        
        # Remove punctuation for comparison
        import string
        original_clean = original_norm.translate(str.maketrans('', '', string.punctuation))
        transcription_clean = transcription_norm.translate(str.maketrans('', '', string.punctuation))
        
        # Calculate similarity
        similarity = difflib.SequenceMatcher(None, original_clean, transcription_clean).ratio()
        return similarity
    
    def check_duration_appropriateness(self, audio_duration: float, text: str) -> bool:
        """Check if audio duration is appropriate for the text length."""
        # Rough estimate: 2-3 words per second for normal speech
        word_count = len(text.split())
        expected_duration = word_count / 2.5  # 2.5 words per second
        
        min_duration = expected_duration * self.min_duration_ratio
        max_duration = expected_duration * self.max_duration_ratio
        
        return min_duration <= audio_duration <= max_duration
    
    def assess_audio_quality(self, audio_path: str) -> Dict:
        """Assess the quality of an audio file."""
        print(f"  Assessing: {Path(audio_path).name}")
        
        # Transcribe audio
        transcription_result = self.transcribe_audio(audio_path)
        if not transcription_result:
            return {
                'satisfactory': False,
                'error': 'Transcription failed',
                'transcription': '',
                'audio_duration': 0
            }
        
        transcription = transcription_result['text']
        audio_duration = transcription_result['duration']
        
        return {
            'satisfactory': True,  # We'll determine this based on comparison
            'transcription': transcription,
            'audio_duration': audio_duration
        }
    
    def find_audio_files(self, dia_dir: str, mooncast_dir: str) -> Dict[int, Dict[str, str]]:
        """Find corresponding DIA and MoonCast audio files."""
        dia_path = Path(dia_dir)
        mooncast_path = Path(mooncast_dir)
        
        audio_files = {}
        
        # Find DIA files
        dia_files = {}
        for f in dia_path.glob("*.wav"):
            # Extract index from filename like "00000.dac_dia_audio.wav"
            parts = f.stem.split('.')
            if len(parts) >= 1:
                dia_files[parts[0]] = f
        
        # Find MoonCast files
        mooncast_files = {}
        for f in mooncast_path.glob("*.wav"):
            # Extract index from filename like "00000.mc_mooncast_audio.wav"
            parts = f.stem.split('.')
            if len(parts) >= 1:
                mooncast_files[parts[0]] = f
        
        # Match files by index
        for index_str in dia_files:
            if index_str in mooncast_files:
                try:
                    index = int(index_str)
                    audio_files[index] = {
                        'dia': str(dia_files[index_str]),
                        'mooncast': str(mooncast_files[index_str])
                    }
                except ValueError:
                    continue
        
        print(f"✅ Found {len(audio_files)} matching audio file pairs")
        return audio_files
    
    def assess_all_pairs(self, sentences: Dict[int, str], audio_files: Dict[int, Dict[str, str]]) -> List[Dict]:
        """Assess all audio pairs and return satisfactory ones."""
        results = []
        satisfactory_pairs = []
        
        print(f"\nAssessing {len(audio_files)} audio pairs...")
        
        for index in sorted(audio_files.keys()):
            if index not in sentences:
                continue
                
            original_text = sentences[index]
            print(f"\n[{index:05d}] Text: {original_text[:50]}...")
            
            # Assess DIA audio
            dia_result = self.assess_audio_quality(audio_files[index]['dia'])
            
            # Assess MoonCast audio
            mooncast_result = self.assess_audio_quality(audio_files[index]['mooncast'])
            
            # Compare transcriptions to see if they say the same thing
            if dia_result['satisfactory'] and mooncast_result['satisfactory']:
                similarity = self.calculate_similarity(dia_result['transcription'], mooncast_result['transcription'])
                # Check if both audio files are less than 6 seconds
                both_short = dia_result['audio_duration'] < 6.0 and mooncast_result['audio_duration'] < 6.0
                both_satisfactory = similarity >= self.min_similarity and both_short
            else:
                similarity = 0.0
                both_satisfactory = False
            
            result = {
                'index': index,
                'original_text': original_text,
                'dia': dia_result,
                'mooncast': mooncast_result,
                'similarity': similarity,
                'both_satisfactory': both_satisfactory
            }
            
            results.append(result)
            
            if both_satisfactory:
                satisfactory_pairs.append(result)
                print(f"  ✅ BOTH SAY THE SAME THING! (similarity: {similarity:.2f}, DIA: {dia_result['audio_duration']:.1f}s, MC: {mooncast_result['audio_duration']:.1f}s)")
            else:
                reason = []
                if similarity < self.min_similarity:
                    reason.append("different content")
                if dia_result['audio_duration'] >= 6.0 or mooncast_result['audio_duration'] >= 6.0:
                    reason.append("too long")
                print(f"  ❌ {' + '.join(reason)} (similarity: {similarity:.2f}, DIA: {dia_result['audio_duration']:.1f}s, MC: {mooncast_result['audio_duration']:.1f}s)")
                print(f"    DIA: '{dia_result['transcription'][:50]}...'")
                print(f"    MC:  '{mooncast_result['transcription'][:50]}...'")
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
        
        return results, satisfactory_pairs
    
    def save_results(self, results: List[Dict], satisfactory_pairs: List[Dict], output_file: str):
        """Save assessment results to JSON file."""
        output_data = {
            'summary': {
                'total_pairs': len(results),
                'satisfactory_pairs': len(satisfactory_pairs),
                'success_rate': len(satisfactory_pairs) / len(results) * 100 if results else 0,
                'criteria': 'Both DIA and MoonCast say similar content AND both audio files < 6 seconds'
            },
            'satisfactory_pairs': satisfactory_pairs
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to {output_file}")
    
    def print_summary(self, results: List[Dict], satisfactory_pairs: List[Dict]):
        """Print assessment summary."""
        print(f"\n{'='*60}")
        print("ASSESSMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total pairs assessed: {len(results)}")
        print(f"Satisfactory pairs: {len(satisfactory_pairs)}")
        print(f"Success rate: {len(satisfactory_pairs)/len(results)*100:.1f}%" if results else "0%")
        print(f"Criteria: Both DIA and MoonCast say similar content AND both audio files < 6 seconds")
        
        if satisfactory_pairs:
            print(f"\nSatisfactory pairs (both DIA and MoonCast work well):")
            for pair in satisfactory_pairs[:10]:  # Show first 10
                dia_dur = pair['dia']['audio_duration']
                mc_dur = pair['mooncast']['audio_duration']
                print(f"  [{pair['index']:05d}] {pair['original_text'][:50]}... (DIA: {dia_dur:.1f}s, MC: {mc_dur:.1f}s)")
            if len(satisfactory_pairs) > 10:
                print(f"  ... and {len(satisfactory_pairs) - 10} more")

def main():
    parser = argparse.ArgumentParser(description="Assess audio quality using Whisper")
    parser.add_argument("--sentences", default="1232.txt", help="File containing original sentences")
    parser.add_argument("--dia-dir", default="dia_audio_output", help="Directory with DIA audio files")
    parser.add_argument("--mooncast-dir", default="mooncast_audio_output", help="Directory with MoonCast audio files")
    parser.add_argument("--output", default="audio_assessment_results.json", help="Output JSON file")
    parser.add_argument("--model-size", default="base", choices=["tiny", "base", "small", "medium", "large"], 
                       help="Whisper model size")
    parser.add_argument("--min-similarity", type=float, default=0.3, help="Minimum text similarity threshold")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    # Initialize assessor
    assessor = AudioQualityAssessor(model_size=args.model_size)
    assessor.min_similarity = args.min_similarity
    
    # Load sentences
    sentences = assessor.load_sentences(args.sentences)
    if not sentences:
        return
    
    # Find audio files
    audio_files = assessor.find_audio_files(args.dia_dir, args.mooncast_dir)
    if not audio_files:
        print("❌ No matching audio files found")
        return
    
    # Limit files if requested
    if args.max_files:
        limited_files = {}
        for i, (index, files) in enumerate(sorted(audio_files.items())):
            if i >= args.max_files:
                break
            limited_files[index] = files
        audio_files = limited_files
        print(f"Limited to {len(audio_files)} files")
    
    # Assess all pairs
    results, satisfactory_pairs = assessor.assess_all_pairs(sentences, audio_files)
    
    # Save results
    assessor.save_results(results, satisfactory_pairs, args.output)
    
    # Print summary
    assessor.print_summary(results, satisfactory_pairs)
    
    print(f"\n✅ Assessment complete! Check {args.output} for detailed results.")

if __name__ == "__main__":
    main()
