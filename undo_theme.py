import os
import glob

replacements_undo = {
    # Backgrounds
    '"#0B1020"': '"#0f172a"',
    '"rgba(255,255,255,0.05)"': '"#1e293b"',
    '"#1F2937"': '"#1e1b4b"',
    
    # Text
    '"#9CA3AF"': '"#94a3b8"',
    '"#E5E7EB"': '"#a5b4fc"',
    
    # Accents
    '"#3B82F6"': '"#6C63FF"',
    '"#06B6D4"': '"#22d3ee"',
    '"#10B981"': '"#34d399"',
}

for filepath in glob.glob('/home/anwar/Desktop/AI_Lab_Studio/modules/*.py'):
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements_undo.items():
        content = content.replace(old, new)
        # Also replace single quotes variations
        content = content.replace(old.replace('"', "'"), new.replace('"', "'"))
        
    with open(filepath, 'w') as f:
        f.write(content)

print("Original theme colors restored across all modules.")
