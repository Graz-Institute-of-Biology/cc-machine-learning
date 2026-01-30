#!/usr/bin/env python3
"""
Dataset Comparison Tool
Compares dataset_v1 with dataset_v1_recreated to verify identical recreation
"""

import hashlib
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime

class DatasetHashComparator:
    def __init__(self):
        self.results = {
            'identical': [],
            'different': [],
            'missing_in_recreated': [],
            'extra_in_recreated': [],
            'statistics': {}
        }
    
    def compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def analyze_pixel_difference(self, file1: Path, file2: Path) -> Dict:
        """Detailed pixel analysis when hashes differ"""
        try:
            img1 = np.array(Image.open(file1))
            img2 = np.array(Image.open(file2))
            
            # Check if shapes differ
            if img1.shape != img2.shape:
                return {
                    'issue': 'dimension_mismatch',
                    'shape_original': img1.shape,
                    'shape_recreated': img2.shape,
                    'diagnosis': 'Cutout mechanism changed (different crop size/location)'
                }
            
            # Calculate pixel differences
            diff = np.abs(img1.astype(float) - img2.astype(float))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            identical_pixels_pct = float(np.sum(diff == 0) / diff.size * 100)
            
            # Diagnose the issue
            diagnosis = []
            if max_diff > 0:
                if max_diff < 5:
                    diagnosis.append('Minor pixel differences (possibly JPEG re-encoding)')
                else:
                    diagnosis.append('Significant pixel differences (different source or cutout location)')
            
            return {
                'issue': 'pixel_mismatch',
                'shape': img1.shape,
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'identical_pixels_percent': identical_pixels_pct,
                'diagnosis': ' | '.join(diagnosis) if diagnosis else 'Different pixel values'
            }
        
        except Exception as e:
            return {
                'issue': 'analysis_error',
                'error': str(e)
            }
    
    def compare_file_pair(self, file1: Path, file2: Path, file_type: str) -> Tuple[bool, Dict]:
        """Compare two files by hash and analyze if different"""
        hash1 = self.compute_file_hash(file1)
        hash2 = self.compute_file_hash(file2)
        
        is_identical = (hash1 == hash2)
        
        details = {
            'filename': file1.name,
            'type': file_type,
            'hash_original': hash1[:16] + '...',
            'hash_recreated': hash2[:16] + '...',
        }
        
        if not is_identical:
            pixel_analysis = self.analyze_pixel_difference(file1, file2)
            details.update(pixel_analysis)
        
        return is_identical, details
    
    def compare_datasets(self, original_dir: Path, recreated_dir: Path):
        """Compare dataset_v1 vs dataset_v1_recreated"""
        original_dir = Path(original_dir)
        recreated_dir = Path(recreated_dir)
        
        # Verify directories exist
        required_dirs = [
            original_dir / 'partial_images',
            original_dir / 'partial_masks',
            recreated_dir / 'partial_images',
            recreated_dir / 'partial_masks'
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Collect all files
        original_images = {f.name: f for f in (original_dir / 'partial_images').glob('*.jpg')}
        recreated_images = {f.name: f for f in (recreated_dir / 'partial_images').glob('*.jpg')}
        
        original_masks = {f.name: f for f in (original_dir / 'partial_masks').glob('*.png')}
        recreated_masks = {f.name: f for f in (recreated_dir / 'partial_masks').glob('*.png')}
        
        # Also check for jpeg extension
        original_images.update({f.name: f for f in (original_dir / 'partial_images').glob('*.jpeg')})
        recreated_images.update({f.name: f for f in (recreated_dir / 'partial_images').glob('*.jpeg')})
        
        print(f"\n📁 Found files:")
        print(f"   Original: {len(original_images)} images, {len(original_masks)} masks")
        print(f"   Recreated: {len(recreated_images)} images, {len(recreated_masks)} masks")
        
        # Find common and missing files
        common_images = set(original_images.keys()) & set(recreated_images.keys())
        common_masks = set(original_masks.keys()) & set(recreated_masks.keys())
        
        missing_images = set(original_images.keys()) - set(recreated_images.keys())
        missing_masks = set(original_masks.keys()) - set(recreated_masks.keys())
        extra_images = set(recreated_images.keys()) - set(original_images.keys())
        extra_masks = set(recreated_masks.keys()) - set(original_masks.keys())
        
        self.results['missing_in_recreated'] = sorted(list(missing_images | missing_masks))
        self.results['extra_in_recreated'] = sorted(list(extra_images | extra_masks))
        
        print(f"\n🔍 Comparing datasets...")
        print(f"   Images to compare: {len(common_images)}")
        print(f"   Masks to compare:  {len(common_masks)}")
        
        if missing_images or missing_masks:
            print(f"   ⚠️  Missing in recreated: {len(missing_images)} images, {len(missing_masks)} masks")
        if extra_images or extra_masks:
            print(f"   ⚠️  Extra in recreated: {len(extra_images)} images, {len(extra_masks)} masks")
        
        # Compare images
        print(f"\n📸 Hashing and comparing {len(common_images)} images...")
        for i, img_name in enumerate(sorted(common_images), 1):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(common_images)}")
            
            is_identical, details = self.compare_file_pair(
                original_images[img_name],
                recreated_images[img_name],
                'image'
            )
            
            if is_identical:
                self.results['identical'].append(details)
            else:
                self.results['different'].append(details)
                print(f"   ❌ {img_name}: {details.get('diagnosis', 'Different')}")
        
        # Compare masks
        print(f"\n🎭 Hashing and comparing {len(common_masks)} masks...")
        for i, mask_name in enumerate(sorted(common_masks), 1):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(common_masks)}")
            
            is_identical, details = self.compare_file_pair(
                original_masks[mask_name],
                recreated_masks[mask_name],
                'mask'
            )
            
            if is_identical:
                self.results['identical'].append(details)
            else:
                self.results['different'].append(details)
                print(f"   ❌ {mask_name}: {details.get('diagnosis', 'Different')}")
        
        # Calculate statistics
        total_files = len(common_images) + len(common_masks)
        identical_count = len([r for r in self.results['identical']])
        different_count = len(self.results['different'])
        
        self.results['statistics'] = {
            'total_compared': total_files,
            'identical': identical_count,
            'different': different_count,
            'match_percentage': (identical_count / total_files * 100) if total_files > 0 else 0,
            'images_compared': len(common_images),
            'masks_compared': len(common_masks),
            'missing_in_recreated': len(self.results['missing_in_recreated']),
            'extra_in_recreated': len(self.results['extra_in_recreated']),
            'comparison_date': datetime.now().isoformat()
        }
        
        return self.results
    
    def print_summary(self):
        """Print comprehensive comparison summary"""
        stats = self.results['statistics']
        
        print("\n" + "="*70)
        print("📊 DATASET COMPARISON SUMMARY")
        print("="*70)
        
        print(f"\n✅ MATCH RATE: {stats['match_percentage']:.2f}%")
        print(f"   • Identical files: {stats['identical']} / {stats['total_compared']}")
        print(f"   • Different files: {stats['different']}")
        
        if stats['different'] == 0 and stats['missing_in_recreated'] == 0 and stats['extra_in_recreated'] == 0:
            print("\n🎉 PERFECT MATCH! Datasets are identical.")
            print("   ✓ Cutout mechanism has NOT changed")
            print("   ✓ All files recreated correctly")
            print("   ✓ Recreation is byte-for-byte identical")
        else:
            print("\n⚠️  DIFFERENCES DETECTED:")
            
            if self.results['different']:
                print(f"\n   Different Files ({len(self.results['different'])}):")
                
                # Group by type
                images_diff = [d for d in self.results['different'] if d['type'] == 'image']
                masks_diff = [d for d in self.results['different'] if d['type'] == 'mask']
                
                if images_diff:
                    print(f"\n   📸 Images ({len(images_diff)}):")
                    for item in images_diff[:10]:
                        print(f"      • {item['filename']}")
                        if 'diagnosis' in item:
                            print(f"        → {item['diagnosis']}")
                        if 'shape_original' in item:
                            print(f"        → Original: {item['shape_original']}, "
                                  f"Recreated: {item['shape_recreated']}")
                        elif 'max_difference' in item:
                            print(f"        → Max pixel diff: {item['max_difference']:.2f}, "
                                  f"Mean: {item['mean_difference']:.2f}, "
                                  f"Identical pixels: {item['identical_pixels_percent']:.2f}%")
                    
                    if len(images_diff) > 10:
                        print(f"      ... and {len(images_diff) - 10} more")
                
                if masks_diff:
                    print(f"\n   🎭 Masks ({len(masks_diff)}):")
                    for item in masks_diff[:10]:
                        print(f"      • {item['filename']}")
                        if 'diagnosis' in item:
                            print(f"        → {item['diagnosis']}")
                        if 'shape_original' in item:
                            print(f"        → Original: {item['shape_original']}, "
                                  f"Recreated: {item['shape_recreated']}")
                        elif 'max_difference' in item:
                            print(f"        → Max pixel diff: {item['max_difference']:.2f}")
                    
                    if len(masks_diff) > 10:
                        print(f"      ... and {len(masks_diff) - 10} more")
            
            if self.results['missing_in_recreated']:
                print(f"\n   ❌ Missing in recreated dataset ({len(self.results['missing_in_recreated'])}):")
                for filename in self.results['missing_in_recreated'][:10]:
                    print(f"      • {filename}")
                if len(self.results['missing_in_recreated']) > 10:
                    print(f"      ... and {len(self.results['missing_in_recreated']) - 10} more")
            
            if self.results['extra_in_recreated']:
                print(f"\n   ➕ Extra files in recreated dataset ({len(self.results['extra_in_recreated'])}):")
                for filename in self.results['extra_in_recreated'][:10]:
                    print(f"      • {filename}")
                if len(self.results['extra_in_recreated']) > 10:
                    print(f"      ... and {len(self.results['extra_in_recreated']) - 10} more")
            
            # Provide diagnosis
            print("\n🔬 DIAGNOSIS:")
            if any('dimension_mismatch' in str(d.get('issue', '')) for d in self.results['different']):
                print("   ⚠️  Different image dimensions detected")
                print("   → Cutout mechanism CHANGED (different crop sizes/locations)")
            if any('pixel_mismatch' in str(d.get('issue', '')) for d in self.results['different']):
                print("   ⚠️  Pixel-level differences detected")
                print("   → Possible causes:")
                print("      • Different source images used")
                print("      • Cutout location/coordinates changed")
                print("      • JPEG re-encoding occurred (should not happen if cutouts are identical)")
        
        print("\n" + "="*70)
    
    def save_report(self, output_file='comparison_report.json'):
        """Save detailed comparison report"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Detailed report saved to: {output_file}")


def main():
    """Main execution function"""
    print("="*70)
    print("🔬 Dataset Recreation Verification Tool")
    print("="*70)
    print("\nComparing: dataset_v1 ↔ dataset_v1_recreated")
    
    try:
        comparator = DatasetHashComparator()
        
        results = comparator.compare_datasets(
            original_dir=r'C:\Users\faulhamm\Documents\Philipp\training\datasets\ATTO\dataset_v11_TF_split',
            recreated_dir= r'C:\Users\faulhamm\Documents\Philipp\training\datasets\ATTO\restore\dataset_v11_TF_split'
        )
        
        comparator.print_summary()
        comparator.save_report('dataset_comparison_report.json')
        
        # Return exit code based on results
        if results['statistics']['different'] == 0 and \
           results['statistics']['missing_in_recreated'] == 0 and \
           results['statistics']['extra_in_recreated'] == 0:
            print("\n✅ SUCCESS: Datasets are identical!")
            return 0
        else:
            print("\n⚠️  WARNING: Datasets have differences!")
            return 1
            
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure the following directories exist:")
        print("  • dataset_v1/images")
        print("  • dataset_v1/masks")
        print("  • dataset_v1_recreated/images")
        print("  • dataset_v1_recreated/masks")
        return 2
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit(main())