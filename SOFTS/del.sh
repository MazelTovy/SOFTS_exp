#!/bin/bash

# Check if a file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 filename"
  exit 1
fi

# Use sed to delete lines starting with "Namespace"
sed -i '/^Namespace/d' "$1"

echo "Lines starting with 'Namespace' have been removed from $1."