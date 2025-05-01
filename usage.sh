#!/bin/bash

# Log file path
LOG_FILE="/var/log/system_usage.log"

echo "Time,CPU Usage (%),Used Memory (GB),GPU Usage (%),GPU Used Memory (GB),GPU Power (W)"

# Infinite loop to log metrics every minute
while true
do
    # Get current date and time
    DATE=$(date '+%Y-%m-%d %H:%M:%S')

    # Get used memory in MiB
    USED_MEM=$(top -b -n1 | grep "MiB Mem" | awk '{printf "%.2f", $8/1024}')

    # Get CPU usage (user + system)
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}')

    # Get GPU usage, memory usage, and power consumption from nvidia-smi
    GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits)

    # Parse GPU stats into variables
    GPU_UTIL=$(echo $GPU_STATS | awk -F, '{print $1}')
    GPU_MEM_USED=$(echo $GPU_STATS | awk -F, '{print $2}')
    GPU_MEM_USED_TRIMMED=$(echo $GPU_MEM_USED | xargs)
    #GPU_MEM_USED_GB=$(echo "scale=2; $GPU_MEM_USED / 1024" | bc)
    GPU_POWER=$(echo $GPU_STATS | awk -F, '{print $3}')
    GPU_POWER_TRIMMED=$(echo $GPU_POWER | xargs)

    # Append the date, CPU usage, used memory, and GPU stats to the log file in one line
    #echo "$DATE, $CPU_USAGE%, $USED_MEM GB, $GPU_UTIL%, $GPU_MEM_USED MB,$GPU_POWER W"
    echo "$DATE,$CPU_USAGE,$USED_MEM,$GPU_UTIL,$GPU_MEM_USED_TRIMMED,$GPU_POWER_TRIMMED"

    # Wait for 60 seconds before repeating
    sleep 1
done
