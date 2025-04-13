#!/bin/bash

# Check input
if [ -z "$1" ]; then
  echo "Usage: $0 input_filename"
  exit 1
fi

input_file="$1"
filename=$(basename "$input_file")
output_file="./$filename"

# Extract lines starting with "mse:0" and save to a new file
grep '^mse:0' "$input_file" > "$output_file"

echo "Extracted lines saved to $output_file"