import subprocess

command = [
    'xvfb-run', 
    'blender',
    '--background',
    './data/blend_files/simple_skinned_character.blend',
    '--render-output',
    './data/renders/simple_skinned_character',
    '--render-frame',
    '1',
]

subprocess.run(command)