#!/bin/bash
#
# compress-dir.sh - Creates a .tar.gz archive from directory contents
#
# Usage: ./compress-dir.sh <directory_path> [output_filename]

# Check if a directory argument was provided
if [ $# -lt 1 ]; then
    echo "Error: No directory specified"
    echo "Usage: $0 <directory_path> [output_filename]"
    exit 1
fi

# Get the directory path and remove trailing slash if present
DIR_PATH="${1%/}"

# Check if the directory exists
if [ ! -d "$DIR_PATH" ]; then
    echo "Error: '$DIR_PATH' is not a valid directory"
    exit 1
fi

# Get the basename of the directory for the output filename
DIR_NAME=$(basename "$DIR_PATH")

# If a second argument is provided, use it as the output filename
# Otherwise, use the directory name
if [ $# -ge 2 ]; then
    OUTPUT_FILE="$2"
else
    OUTPUT_FILE="${DIR_NAME}.tar.xz"
fi

# Create the tar.gz file
echo "Compressing contents of '$DIR_PATH' into '$OUTPUT_FILE'..."
tar -cJf "$OUTPUT_FILE" -C "$DIR_PATH" .

# Check if the tar command was successful
if [ $? -eq 0 ]; then
    echo "Compression completed successfully"
    echo "Archive created: $OUTPUT_FILE ($(du -h "$OUTPUT_FILE" | cut -f1))"
    exit 0
else
    echo "Error: Compression failed"
    exit 1
fi