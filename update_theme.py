import os
import glob

replacements = {
    # Backgrounds
    '"#0f172a"': '"#0B1020"',
    '"#111827"': '"#111827"', # Stays the same 
    '"#1e293b"': '"rgba(255,255,255,0.05)"',
    '"#1e1b4b"': '"#1F2937"',
    
    # Text
    '"#94a3b8"': '"#9CA3AF"',
    '"#a5b4fc"': '"#E5E7EB"',
    
    # Accents
    '"#6C63FF"': '"#3B82F6"',
    '"#22d3ee"': '"#06B6D4"',
    '"#34d399"': '"#10B981"',
}

for filepath in glob.glob('/home/anwar/Desktop/AI_Lab_Studio/modules/*.py'):
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
        # Also replace single quotes variations
        content = content.replace(old.replace('"', "'"), new.replace('"', "'"))
        
    with open(filepath, 'w') as f:
        f.write(content)

print("Theme colors updated across all modules.")
