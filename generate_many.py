import argparse
import subprocess
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_text = 'The generation script.'
    parser.add_argument('--script', type=str)
    help_text = 'Number of generations.'
    parser.add_argument('--number', type=int, default=1)
    args = parser.parse_args()

    command = [
        'xvfb-run', 
        'blender',
        '--background',
        '-P',
        'scripts/' + args.script
    ]

    for i in range(args.number):
        subprocess.run(command)
        os.rename('data/renders', f'data/generation_{i}')
        os.mkdir('data/renders')
