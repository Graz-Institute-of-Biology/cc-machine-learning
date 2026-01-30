#!/usr/bin/env python3
import re
import sys

def extract_epoch_iou(filename):
    """
    Extract epoch numbers and their corresponding validation IoU scores.
    
    The pattern in the file is:
    - validation line with iou_score
    - epoch line with epoch number
    """
    
    results = []
    current_iou = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Check for validation line with iou_score
            if 'valid:' in line and 'iou_score' in line:
                # Extract iou_score value
                iou_match = re.search(r'iou_score\s*-\s*([0-9.]+)', line)
                if iou_match:
                    current_iou = float(iou_match.group(1))
            
            # Check for epoch line
            elif 'Epoch:' in line and current_iou is not None:
                # Extract epoch number (the first number after "Epoch:")
                epoch_match = re.search(r'Epoch:\s*(\d+)', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    results.append((epoch, current_iou))
                    current_iou = None  # Reset for next epoch
    
    return results

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python extract_metrics.py <logfile>")
    #     sys.exit(1)
    
    filename = "slurm-4389361.out"
    results = extract_epoch_iou(filename)
    
    # Print results
    print("Epoch\tIoU Score")
    print("-" * 25)
    for epoch, iou in results:
        print(f"{epoch}\t{iou:.4f}")
    
    # Also save to CSV
    output_file = "epoch_iou_scores.csv"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Epoch,IoU_Score\n")
        for epoch, iou in results:
            f.write(f"{epoch},{iou:.4f}\n")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total epochs found: {len(results)}")

if __name__ == "__main__":
    main()