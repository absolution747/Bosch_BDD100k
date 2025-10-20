#!/bin/bash

# --- Configuration ---
OUTPUT_FILENAME="bdd100k_data.zip"
# Set the directory where the unzipped files will go (data directory in this case)
UNZIP_DIR="/workspace/data" 

# ---1 Argument Check ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <GOOGLE_DRIVE_FILE_ID>"
    echo "Example: $0 <32 char string>"
    exit 1
fi

FILE_ID="$1"

# Check and install 'unzip' utility
if ! command -v unzip &> /dev/null; then
    echo "1. Installing 'unzip' utility..."
    sudo apt update && sudo apt install unzip -y
else
    echo "1. 'unzip' utility is already installed."
fi

# --- 2. Download the File ---
echo "2. Starting download for File ID: $FILE_ID (Saving as: $OUTPUT_FILENAME)"

# Use gdown to download the file using the passed ID
gdown --id "$FILE_ID" -O "$OUTPUT_FILENAME" --fuzzy

# --- 3. Verification, Unzip, and Cleanup ---
if [ -f "$OUTPUT_FILENAME" ]; then
    echo "------------------------------------------------"
    echo "✅ Download successful!"
    echo "File size: $(du -h "$OUTPUT_FILENAME" | cut -f1)"
    echo "------------------------------------------------"

    # --- NEW STEP: Unzip the File ---
    echo "3. Unzipping $OUTPUT_FILENAME into $UNZIP_DIR..."
    unzip "$OUTPUT_FILENAME" -d "$UNZIP_DIR"

    # Check if unzip was successful (exit code 0)
    if [ "$?" -eq 0 ]; then
        echo "✅ Unzip successful!"

        # --- NEW STEP: Delete the Zip File ---
        echo "4. Deleting $OUTPUT_FILENAME..."
        rm "$OUTPUT_FILENAME"
        echo "✅ Cleanup successful. $OUTPUT_FILENAME deleted."
    else
        echo "❌ ERROR: Unzip failed. The zip file remains for inspection."
    fi

else
    echo "------------------------------------------------"
    echo "❌ ERROR: gdown failed. Check Google Drive permissions or File ID."
    echo "------------------------------------------------"
fi