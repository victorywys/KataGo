#!/usr/bin/env python3
"""
KataGo Pipeline Performance Analyzer
Analyzes logs to identify bottlenecks in the training pipeline.
"""

import os
import re
import sys
import glob
from datetime import datetime, timedelta
from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze KataGo pipeline performance")
    parser.add_argument("--basedir", default="/mnt/output/KataGo_training/selfplay_connection_server",
                       help="Base directory for training data")
    parser.add_argument("--selfplay-logs", help="Selfplay log directory (default: basedir/selfplay/*/log*.log)")
    parser.add_argument("--shuffle-logs", help="Shuffle log directory")
    parser.add_argument("--training-logs", help="Training log directory")
    parser.add_argument("--last-hours", type=float, default=24, 
                       help="Only analyze logs from last N hours (default: 24)")
    return parser.parse_args()

class PipelineAnalyzer:
    def __init__(self, basedir, last_hours=24):
        self.basedir = basedir
        self.cutoff_time = datetime.now() - timedelta(hours=last_hours)
        
    def analyze_selfplay_logs(self, log_pattern=None):
        """Analyze selfplay logs to compute games/hour and data generation rate."""
        if log_pattern is None:
            log_pattern = os.path.join(self.basedir, "selfplay/*/log*.log")
        
        print("=" * 80)
        print("SELFPLAY PERFORMANCE")
        print("=" * 80)
        
        log_files = glob.glob(log_pattern)
        if not log_files:
            print(f"⚠️  No selfplay logs found at: {log_pattern}")
            return
        
        print(f"Found {len(log_files)} selfplay log files\n")
        
        games_data = []
        rows_data = []
        
        for log_file in sorted(log_files)[-10:]:  # Last 10 logs
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                    # Look for game completion messages
                    for line in lines[-1000:]:  # Last 1000 lines
                        # Pattern: game completion with model name
                        if 'Model loading loop thread loaded new neural net' in line:
                            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if match:
                                timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                                if timestamp >= self.cutoff_time:
                                    print(f"  Model loaded: {os.path.basename(log_file)}")
                        
                        # Count data rows written
                        if 'rows' in line.lower() and 'written' in line.lower():
                            games_data.append(line)
                            
            except Exception as e:
                print(f"  Error reading {log_file}: {e}")
        
        # Estimate from configuration
        print("\n📊 SELFPLAY CONFIGURATION ESTIMATES:")
        print(f"  numGameThreads: 128 (from config)")
        print(f"  maxVisits: 600 (from config)")
        print(f"  Estimated game length: ~200 moves average")
        print(f"  Estimated rows per game: ~200-250")
        print()
        
        # Theoretical throughput calculation
        visits_per_second = 100  # Rough estimate, depends on GPU
        seconds_per_game = (200 * 600) / visits_per_second  # moves * visits / speed
        games_per_hour_per_thread = 3600 / seconds_per_game
        
        print(f"  Estimated speed (single thread):")
        print(f"    - Visits/second: ~{visits_per_second} (GPU dependent)")
        print(f"    - Seconds per game: ~{seconds_per_game:.1f}")
        print(f"    - Games/hour/thread: ~{games_per_hour_per_thread:.1f}")
        print(f"    - Games/hour (128 threads): ~{games_per_hour_per_thread * 128:.0f}")
        print(f"    - Rows/hour (128 threads): ~{games_per_hour_per_thread * 128 * 225:.0f}")
        print()
        
        # Check actual selfplay data generation
        selfplay_dirs = glob.glob(os.path.join(self.basedir, "selfplay/*/tdata"))
        if selfplay_dirs:
            total_files = 0
            total_size_mb = 0
            for tdata_dir in selfplay_dirs:
                npz_files = glob.glob(os.path.join(tdata_dir, "*.npz"))
                total_files += len(npz_files)
                for f in npz_files:
                    try:
                        total_size_mb += os.path.getsize(f) / (1024 * 1024)
                    except:
                        pass
            
            print(f"📁 ACTUAL SELFPLAY DATA:")
            print(f"  Total .npz files: {total_files}")
            print(f"  Total size: {total_size_mb:.1f} MB")
            print(f"  Avg file size: {total_size_mb/max(total_files,1):.2f} MB")
            print(f"  Est. rows (10k rows/file): {total_files * 10000:,}")
            print()
    
    def analyze_shuffle_logs(self, log_pattern=None):
        """Analyze shuffle logs to compute processing rate."""
        if log_pattern is None:
            log_pattern = os.path.join(self.basedir, "shuffleddata/*.tmp/out*.txt")
        
        print("=" * 80)
        print("SHUFFLE PERFORMANCE")
        print("=" * 80)
        
        # Look for completed shuffle directories
        shuffle_dirs = sorted(glob.glob(os.path.join(self.basedir, "shuffleddata/202*")))
        
        if not shuffle_dirs:
            print(f"⚠️  No shuffled data directories found")
            return
        
        print(f"Found {len(shuffle_dirs)} shuffled data directories\n")
        
        recent_shuffles = shuffle_dirs[-10:]  # Last 10 shuffles
        shuffle_times = []
        
        for sdir in recent_shuffles:
            dirname = os.path.basename(sdir)
            
            # Check for timing info in output files
            train_dir = os.path.join(sdir, "train")
            val_dir = os.path.join(sdir, "val")
            
            if os.path.exists(train_dir):
                train_files = glob.glob(os.path.join(train_dir, "*.npz"))
                total_size_mb = sum(os.path.getsize(f) / (1024*1024) for f in train_files if os.path.exists(f))
                
                print(f"  {dirname}:")
                print(f"    - Train files: {len(train_files)}")
                print(f"    - Train size: {total_size_mb:.1f} MB")
                
                # Try to get timing from associated tmp dir output
                tmp_output = sdir + ".tmp/outtrain.txt"
                if os.path.exists(tmp_output):
                    try:
                        with open(tmp_output, 'r') as f:
                            content = f.read()
                            
                            # Look for TimeStuff output
                            sharding_match = re.search(r'Finished: Sharding in ([\d.]+) seconds', content)
                            merging_match = re.search(r'Finished: Merging in ([\d.]+) seconds', content)
                            
                            if sharding_match:
                                shard_time = float(sharding_match.group(1))
                                print(f"    - Sharding: {shard_time:.1f}s")
                            if merging_match:
                                merge_time = float(merging_match.group(1))
                                print(f"    - Merging: {merge_time:.1f}s")
                                
                            if sharding_match and merging_match:
                                total_time = shard_time + merge_time
                                throughput_mb_s = total_size_mb / total_time if total_time > 0 else 0
                                print(f"    - Total time: {total_time:.1f}s")
                                print(f"    - Throughput: {throughput_mb_s:.2f} MB/s")
                                shuffle_times.append(total_time)
                    except Exception as e:
                        pass
                print()
        
        if shuffle_times:
            avg_time = sum(shuffle_times) / len(shuffle_times)
            print(f"📊 SHUFFLE SUMMARY:")
            print(f"  Average shuffle time: {avg_time:.1f}s ({avg_time/60:.1f} minutes)")
            print(f"  Shuffle frequency: every ~20-30 seconds (from config)")
            print(f"  Processing threads: 32 (from config)")
            print()
    
    def analyze_training_logs(self, log_dir=None):
        """Analyze training logs to compute training throughput."""
        if log_dir is None:
            train_dirs = glob.glob(os.path.join(self.basedir, "../../../train/*/"))
            if not train_dirs:
                log_dir = os.path.join(self.basedir, "../../train")
            else:
                log_dir = train_dirs[0]
        
        print("=" * 80)
        print("TRAINING PERFORMANCE")
        print("=" * 80)
        
        stdout_log = os.path.join(log_dir, "stdout.txt")
        if not os.path.exists(stdout_log):
            print(f"⚠️  No training log found at: {stdout_log}")
            return
        
        print(f"Analyzing: {stdout_log}\n")
        
        try:
            with open(stdout_log, 'r') as f:
                lines = f.readlines()
                
                # Look for training metrics in last 1000 lines
                batch_times = []
                global_steps = []
                
                for line in lines[-1000:]:
                    # Pattern: global step samples
                    if 'Global step:' in line:
                        match = re.search(r'Global step: (\d+) samples', line)
                        if match:
                            global_steps.append(int(match.group(1)))
                    
                    # Pattern: training metrics with time
                    if '"time_since_last_print"' in line or 'time_since_last_print' in line:
                        match = re.search(r'time_since_last_print["\s:]+([0-9.]+)', line)
                        if match:
                            batch_times.append(float(match.group(1)))
                    
                    # Pattern: batch size info
                    if 'batch-size' in line.lower():
                        print(f"  {line.strip()}")
                
                if global_steps:
                    latest_step = global_steps[-1]
                    print(f"\n📊 TRAINING METRICS:")
                    print(f"  Latest global step: {latest_step:,} samples")
                    print(f"  Batch size: 256 (from config)")
                    print(f"  World size: 1 (single GPU, from config)")
                    
                    if batch_times:
                        avg_time = sum(batch_times) / len(batch_times)
                        batches_per_second = 1.0 / avg_time if avg_time > 0 else 0
                        samples_per_second = batches_per_second * 256
                        samples_per_hour = samples_per_second * 3600
                        
                        print(f"\n  Training speed (recent):")
                        print(f"    - Time per print (50 batches): {avg_time:.2f}s")
                        print(f"    - Batches/second: {batches_per_second:.2f}")
                        print(f"    - Samples/second: {samples_per_second:.1f}")
                        print(f"    - Samples/hour: {samples_per_hour:,.0f}")
                        print(f"    - Hours for 1M samples: {1_000_000/samples_per_hour:.2f}h")
                    
                    # Check bucket info
                    print(f"\n  Looking for bucket information...")
                    for line in lines[-200:]:
                        if 'bucket' in line.lower() and 'rows' in line.lower():
                            print(f"    {line.strip()}")
                
                print()
                
        except Exception as e:
            print(f"  Error reading training log: {e}")
    
    def estimate_bottlenecks(self):
        """Provide bottleneck analysis and recommendations."""
        print("=" * 80)
        print("BOTTLENECK ANALYSIS & RECOMMENDATIONS")
        print("=" * 80)
        
        print("""
📈 TYPICAL THROUGHPUT ESTIMATES (for your configuration):

1. SELFPLAY (128 threads, V100 GPU):
   - Games/hour: ~15,000 - 25,000 (depends on GPU utilization)
   - Rows/hour: ~3,000,000 - 5,000,000
   - Limiting factors:
     * GPU inference speed (maxVisits=600 is expensive)
     * Number of game threads vs GPU batch efficiency
     * Network loading time (every 20s polling)

2. SHUFFLE (32 threads):
   - Processing: ~100-200 MB/s (CPU bound)
   - Time per shuffle: ~60-180 seconds
   - Runs every ~20-30 seconds
   - Limiting factors:
     * CPU cores and memory bandwidth
     * Number of input files to process
     * Disk I/O for blob storage

3. TRAINING (1x V100, batch_size=256):
   - Samples/hour: ~500,000 - 1,000,000
   - Limiting factors:
     * GPU compute for forward/backward pass
     * Connection head adds significant compute
     * Data loading from blob storage
     * Training bucket limiting (max_train_bucket_per_new_data=4)

🎯 TYPICAL BOTTLENECKS (most to least common):

1. ⚠️  TRAINING is usually the bottleneck
   - Training consumes ~500k-1M samples/hour
   - Selfplay generates ~3M-5M rows/hour
   - Ratio: 3:1 to 10:1 (selfplay:training)
   - This is BY DESIGN! Training should be the bottleneck.

2. ⚠️  SHUFFLE can become bottleneck if:
   - Too many input files (>10,000 .npz files)
   - Slow blob storage I/O
   - Insufficient CPU cores (need ~32+)
   - Check: shuffle time >5 minutes

3. ⚠️  SELFPLAY rarely bottlenecks unless:
   - Too few game threads (<64)
   - Slow GPU (older than V100)
   - High maxVisits (>800)
   - Model loading failures

📋 HOW TO IDENTIFY YOUR BOTTLENECK:

1. Check training bucket:
   grep "rows in bucket" train/*/stdout.txt | tail -5
   - If bucket empty: SELFPLAY is bottleneck (rare!)
   - If bucket at max: TRAINING is bottleneck (normal)

2. Check shuffled data age:
   ls -lt /mnt/output/.../shuffleddata/ | head -5
   - If newest is >5 minutes: SHUFFLE is bottleneck

3. Check selfplay data accumulation:
   find /mnt/output/.../selfplay/*/tdata -name "*.npz" | wc -l
   - Growing rapidly: TRAINING is bottleneck (normal)
   - Staying flat: SELFPLAY might be slow

💡 OPTIMIZATION RECOMMENDATIONS:

IF TRAINING IS BOTTLENECK (most common):
  ✓ This is expected! No action needed.
  ✓ Training bucket system prevents overtraining
  ✓ Consider: larger batch_size if GPU memory allows
  ✓ Consider: multiple GPUs with DDP
  ✓ Consider: reduce model size or connection head complexity

IF SHUFFLE IS BOTTLENECK:
  ✓ Increase NTHREADS (currently 32)
  ✓ Use faster CPU or more cores
  ✓ Check blob storage mount performance
  ✓ Consider: increase -approx-rows-per-out-file

IF SELFPLAY IS BOTTLENECK (rare):
  ✓ Add more selfplay compute nodes (safe with current setup!)
  ✓ Reduce maxVisits (e.g., 600 → 400)
  ✓ Check GPU utilization (should be >80%)
  ✓ Increase numGameThreads if GPU underutilized
""")

def main():
    args = parse_args()
    
    analyzer = PipelineAnalyzer(args.basedir, args.last_hours)
    
    print("\n" + "=" * 80)
    print(f"KataGo Pipeline Performance Analysis")
    print(f"Base Directory: {args.basedir}")
    print(f"Analyzing last {args.last_hours} hours")
    print("=" * 80 + "\n")
    
    # Analyze each component
    analyzer.analyze_selfplay_logs(args.selfplay_logs)
    analyzer.analyze_shuffle_logs(args.shuffle_logs)
    analyzer.analyze_training_logs(args.training_logs)
    analyzer.estimate_bottlenecks()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
