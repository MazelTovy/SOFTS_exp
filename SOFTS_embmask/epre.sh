#!/bin/bash

# Check input
if [ -z "$1" ]; then
  echo "Usage: $0 input_filename"
  exit 1
fi

input_file="$1"
filename=$(basename "$input_file")
temp_file="./tmp_filtered_$filename"
output_file="./prefixed_$filename"

# Step 1: Extract lines starting with "mse:0"
grep '^mse:0' "$input_file" > "$temp_file"

# Step 2: Apply prefix logic
# Parameters
numbers=("96_96" "96_192" "96_336" "96_720")
datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2")

# Read filtered lines into array
mapfile -t lines < "$temp_file"

# Safety check
if [ "${#lines[@]}" -ne 80 ]; then
  echo "Warning: Expected 80 lines after filtering, got ${#lines[@]}"
fi

# Clear output file
> "$output_file"

# Initialize counter
counter=0

# Prefix each line as <dataset>_<number>
for i in {1..5}; do
  for dataset in "${datasets[@]}"; do
    for number in "${numbers[@]}"; do
      prefix="${dataset}_${number}"
      if [ $counter -lt ${#lines[@]} ]; then
        echo "${prefix} ${lines[$counter]}" >> "$output_file"
        ((counter++))
      fi
    done
  done
done

# Clean up
rm "$temp_file"

echo "Prefixed and filtered lines saved to $output_file"