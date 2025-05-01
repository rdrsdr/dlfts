#!/bin/bash

start_time=$(date +%s)  # Capture start time
"$@"                     # Execute the command
end_time=$(date +%s)    # Capture end time

elapsed=$((end_time - start_time))  # Calculate elapsed time

echo "Execution time: $elapsed seconds"
