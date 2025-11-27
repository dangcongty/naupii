#!/bin/bash

SOURCE="/home/khoi/PycharmProjects/naupii/app.py"
TARGET_DIR="/home/otanics/naupii"
TARGET="otanics@192.168.10.81:$TARGET_DIR/"

SOURCE1="/home/khoi/PycharmProjects/naupii/templates/stream.html"
TARGET_DIR1="/home/otanics/naupii/templates"
TARGET1="otanics@192.168.10.81:$TARGET_DIR1/"

scp "$SOURCE" "$TARGET"
scp "$SOURCE1" "$TARGET1"

echo "Copied $SOURCE -> $TARGET"
echo "Copied $SOURCE1 -> $TARGET1"
