import os
import re
import sys

# Configuration
LIMIT = 500
changes_made = 0
modified_dirs = set()  # Tracks unique directories that had modifications

repo_path = './demonstrations_v2'

# Regex patterns
import_pattern = re.compile(r'import\s+pennylane\s+as\s+qml\b')
alias_pattern = re.compile(r'\bqml\.')

for root, dirs, files in os.walk(repo_path):
    # Sort directories and files in-place so the traversal is alphabetical
    dirs.sort()
    files.sort()
    
    # Optional: skip hidden directories like .git or virtual environments
    if '.git' in root or 'venv' in root:
        continue
        
    for file in files:
        if changes_made >= LIMIT:
            break
            
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            new_lines = []
            file_modified = False
            
            for line in lines:
                original_line = line
                
                # Apply both regex replacements
                line = import_pattern.sub('import pennylane as qp', line)
                line = alias_pattern.sub('qp.', line)
                
                new_lines.append(line)
                
                if original_line != line:
                    file_modified = True
                    changes_made += 1
                    
                if changes_made >= LIMIT:
                    # If limit reached mid-file, keep the rest of the file unchanged
                    new_lines.extend(lines[len(new_lines):])
                    break
                    
            if file_modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                
                # Add the current directory to our set of modified directories
                modified_dirs.add(root)
                
    if changes_made >= LIMIT:
        print(f"Reached limit of {LIMIT} line changes. Stopping.")
        break

print(f"\nRefactoring complete. Total lines changed: {changes_made}")

# Print the summary of modified subdirectories
if modified_dirs:
    print("Files were modified in the following directories:")
    for directory in sorted(modified_dirs):
        print(f"  - {directory}")
else:
    print("No directories were modified (no matches found).")