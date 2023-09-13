import subprocess

command = [
    'xvfb-run', 
    'blender',
    '--background',
    '-P',
    'scripts/generate_chubby.py'
]

subprocess.run(command)