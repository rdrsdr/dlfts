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
rm timing2.txt
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 12 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 13 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 14 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 15 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 16 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 17 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 18 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 19 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 20 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 21 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
python predict.py 22 "$workdir" "$file" "$date"
echo "finish;$(date +"%Y-%m-%d %H:%M:%S")" >> timing2.txt
