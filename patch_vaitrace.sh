#!/bin/bash

############################################################################
# Copyright 2026 GradeBuilder SL                                           #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

# Patch vaitrace to fix KeyError bug in function symbol mapping

set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script requires root privileges to patch vaitrace. Running with sudo..."
    exec sudo "$0" "$@"
fi

VAITRACE_FILE="/usr/bin/xlnx/vaitrace/tracer/function.py"

echo "============================================================"
echo "Patching vaitrace to fix KeyError bug"
echo "============================================================"
echo ""

# Check if file exists
if [ ! -f "$VAITRACE_FILE" ]; then
    echo "Error: $VAITRACE_FILE not found!"
    exit 1
fi

# Create backup
BACKUP_FILE="${VAITRACE_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup: $BACKUP_FILE"
cp "$VAITRACE_FILE" "$BACKUP_FILE"

# Check if already patched
if grep -q "# PATCHED: Handle missing function symbols" "$VAITRACE_FILE"; then
    echo ""
    echo "✓ vaitrace is already patched!"
    echo ""
    exit 0
fi

echo "Patching $VAITRACE_FILE..."
echo ""

# Create the patch
# The bug is at line 416: data_out.append(l.replace(ftraceSym, self.ftraceSymMap[ftraceSym]))
# We need to handle the KeyError gracefully

python3 << 'PYTHON_PATCH'
import sys

# Read the file
with open('/usr/bin/xlnx/vaitrace/tracer/function.py', 'r') as f:
    lines = f.readlines()

# Find and patch the problematic line
patched = False
new_lines = []
for i, line in enumerate(lines):
    if 'data_out.append(l.replace(ftraceSym, self.ftraceSymMap[ftraceSym]))' in line and not patched:
        # Get the indentation
        indent = len(line) - len(line.lstrip())
        spaces = ' ' * indent

        # Add patched version with try-except
        new_lines.append(f"{spaces}# PATCHED: Handle missing function symbols gracefully\n")
        new_lines.append(f"{spaces}try:\n")
        new_lines.append(f"{spaces}    data_out.append(l.replace(ftraceSym, self.ftraceSymMap[ftraceSym]))\n")
        new_lines.append(f"{spaces}except KeyError:\n")
        new_lines.append(f"{spaces}    # Symbol not in map, keep original or use placeholder\n")
        new_lines.append(f"{spaces}    data_out.append(l.replace(ftraceSym, ftraceSym))  # Keep original symbol\n")
        patched = True
    else:
        new_lines.append(line)

if not patched:
    print("Error: Could not find the line to patch!")
    sys.exit(1)

# Write back
with open('/usr/bin/xlnx/vaitrace/tracer/function.py', 'w') as f:
    f.writelines(new_lines)

print("✓ Patch applied successfully!")

PYTHON_PATCH

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ vaitrace has been patched successfully!"
    echo "============================================================"
    echo ""
    echo "Backup saved to: $BACKUP_FILE"
    echo ""
    echo "You can now run vaitrace without the KeyError bug."
    echo "If you need to restore the original:"
    echo "  sudo cp $BACKUP_FILE $VAITRACE_FILE"
    echo ""
else
    echo ""
    echo "✗ Patching failed! Restoring backup..."
    cp "$BACKUP_FILE" "$VAITRACE_FILE"
    exit 1
fi
