#!/bin/bash
# Quick pipeline health check script

BASEDIR="${1:-/mnt/output/KataGo_training/selfplay_connection_server}"

echo "=========================================="
echo "KataGo Pipeline Quick Check"
echo "=========================================="
echo "Base Directory: $BASEDIR"
echo ""

# 1. Check selfplay data generation rate
echo "📊 SELFPLAY STATUS:"
echo "-------------------"
SELFPLAY_FILES=$(find "$BASEDIR"/selfplay/*/tdata -name "*.npz" 2>/dev/null | wc -l)
echo "Total selfplay .npz files: $SELFPLAY_FILES"
echo "Estimated rows: $((SELFPLAY_FILES * 10000))"

if [ -d "$BASEDIR/selfplay" ]; then
    NEWEST_SELFPLAY=$(find "$BASEDIR"/selfplay/*/tdata -name "*.npz" -type f 2>/dev/null | head -1)
    if [ -n "$NEWEST_SELFPLAY" ]; then
        LAST_SELFPLAY=$(stat -c %y "$NEWEST_SELFPLAY" 2>/dev/null | cut -d'.' -f1)
        echo "Latest selfplay file: $LAST_SELFPLAY"
    fi
fi
echo ""

# 2. Check shuffle status
echo "🔀 SHUFFLE STATUS:"
echo "-------------------"
SHUFFLE_DIRS=$(ls -dt "$BASEDIR"/shuffleddata/202* 2>/dev/null | head -5)
if [ -n "$SHUFFLE_DIRS" ]; then
    echo "Recent shuffle directories:"
    echo "$SHUFFLE_DIRS" | while read dir; do
        TRAIN_FILES=$(find "$dir/train" -name "*.npz" 2>/dev/null | wc -l)
        SIZE=$(du -sh "$dir/train" 2>/dev/null | cut -f1)
        echo "  $(basename $dir): $TRAIN_FILES files, $SIZE"
    done
    
    LATEST_SHUFFLE=$(echo "$SHUFFLE_DIRS" | head -1)
    LATEST_SHUFFLE_TIME=$(basename "$LATEST_SHUFFLE" | cut -d'-' -f1-2)
    echo "Latest shuffle: $LATEST_SHUFFLE_TIME"
else
    echo "⚠️  No shuffled data found!"
fi
echo ""

# 3. Check training status
echo "🎓 TRAINING STATUS:"
echo "-------------------"
TRAIN_LOG="$BASEDIR/../../train/KataGo_connection_pred/stdout.txt"
if [ -f "$TRAIN_LOG" ]; then
    LATEST_STEP=$(grep "Global step:" "$TRAIN_LOG" | tail -1)
    echo "Latest training step: $LATEST_STEP"
    
    echo ""
    echo "Recent bucket status:"
    grep -i "rows in bucket" "$TRAIN_LOG" | tail -3
    
    echo ""
    echo "Recent data updates:"
    grep "Updated training data:" "$TRAIN_LOG" | tail -3
else
    echo "⚠️  Training log not found at: $TRAIN_LOG"
fi
echo ""

# 4. Check for recent activity
echo "⏱️  RECENT ACTIVITY:"
echo "-------------------"
NOW=$(date +%s)

# Last selfplay activity
if [ -n "$NEWEST_SELFPLAY" ]; then
    SELFPLAY_TIME=$(stat -c %Y "$NEWEST_SELFPLAY" 2>/dev/null)
    SELFPLAY_AGE=$((NOW - SELFPLAY_TIME))
    echo "Selfplay last active: $((SELFPLAY_AGE / 60)) minutes ago"
    if [ $SELFPLAY_AGE -gt 600 ]; then
        echo "  ⚠️  Warning: Selfplay may be stalled (>10 min since last file)"
    fi
fi

# Last shuffle activity
if [ -n "$LATEST_SHUFFLE" ]; then
    if [ -d "$LATEST_SHUFFLE" ]; then
        SHUFFLE_TIME=$(stat -c %Y "$LATEST_SHUFFLE" 2>/dev/null)
        SHUFFLE_AGE=$((NOW - SHUFFLE_TIME))
        echo "Shuffle last active: $((SHUFFLE_AGE / 60)) minutes ago"
        if [ $SHUFFLE_AGE -gt 600 ]; then
            echo "  ⚠️  Warning: Shuffle may be stalled (>10 min since last dir)"
        fi
    fi
fi

# Training activity
if [ -f "$TRAIN_LOG" ]; then
    TRAIN_TIME=$(stat -c %Y "$TRAIN_LOG" 2>/dev/null)
    TRAIN_AGE=$((NOW - TRAIN_TIME))
    echo "Training last active: $((TRAIN_AGE / 60)) minutes ago"
    if [ $TRAIN_AGE -gt 600 ]; then
        echo "  ⚠️  Warning: Training may be stalled (>10 min since log update)"
    fi
fi
echo ""

# 5. Quick bottleneck assessment
echo "🎯 BOTTLENECK ASSESSMENT:"
echo "-------------------------"

# Count rate of data generation vs consumption
if [ $SELFPLAY_FILES -gt 1000 ]; then
    echo "✓ Selfplay generating data actively ($SELFPLAY_FILES files)"
else
    echo "⚠️  Low selfplay data volume ($SELFPLAY_FILES files)"
fi

if [ -n "$SHUFFLE_DIRS" ]; then
    SHUFFLE_COUNT=$(echo "$SHUFFLE_DIRS" | wc -l)
    if [ $SHUFFLE_COUNT -ge 3 ]; then
        echo "✓ Shuffle processing regularly ($SHUFFLE_COUNT recent dirs)"
    else
        echo "⚠️  Few recent shuffles ($SHUFFLE_COUNT dirs)"
    fi
fi

# Check if training bucket is mentioned
if [ -f "$TRAIN_LOG" ]; then
    BUCKET_STATUS=$(grep -i "rows in bucket" "$TRAIN_LOG" | tail -1)
    if echo "$BUCKET_STATUS" | grep -q "0\|low\|empty"; then
        echo "⚠️  Training bucket low - SELFPLAY may be bottleneck (rare!)"
    elif echo "$BUCKET_STATUS" | grep -q "5000000\|max\|cap"; then
        echo "✓ Training bucket at max - TRAINING is bottleneck (expected!)"
    else
        echo "✓ Training bucket active"
    fi
fi

echo ""
echo "=========================================="
echo "Run './analyze_pipeline_performance.py' for detailed analysis"
echo "=========================================="
