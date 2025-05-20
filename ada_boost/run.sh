#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log filename
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="logs/run_${timestamp}.log"

# Run main.py and tee output to both terminal and log file
python main.py | tee "$logfile"

echo "Execution completed. Log saved to $logfile"