#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <workdir> <file> <date>"
    echo "Example: $0 ./datasets/247/daily/ dataset.csv 2024-08-26"
    exit 1
fi

# Assign arguments to variables
workdir=$1
file=$2
date=$3
arg4=$4

# Call the Python script with the same arguments
rm timing.txt
echo "LSTM;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 0 "$workdir" "$file" "$date"
echo "GRU;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 1 "$workdir" "$file" "$date"
echo "MLP;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 2 "$workdir" "$file" "$date"
echo "DLINEAR;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 3 "$workdir" "$file" "$date"
echo "NLINEAR;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 4 "$workdir" "$file" "$date"
echo "INFORMER;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 5 "$workdir" "$file" "$date"
echo "AUTOFORMER;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 6 "$workdir" "$file" "$date"
echo "FEDFORMER;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 7 "$workdir" "$file" "$date"
echo "BITCN;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 8 "$workdir" "$file" "$date"
echo "RNN;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 9 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 10 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
python predict.py 11 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing.txt
