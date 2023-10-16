import argparse
import subprocess
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_text = 'The generation script.'
    parser.add_argument('--script', type=str)
    help_text = 'Number of generations.'
    parser.add_argument('--number', type=int, default=1)
    parser.add_argument('--save_directory', type=str, default='data/renders')
    parser.add_argument('--num_renders', type=int, default=200)
    args = parser.parse_args()

    if not os.path.exists(args.save_directory):
        os.mkdir(args.save_directory)

    command = [
        'xvfb-run', 
        'blender',
        '--background',
        '-P',
        f'scripts/pcg/{args.script}',
        '--',
        '--save_directory',
        args.save_directory,
        '--num_renders',
        str(args.num_renders)
    ]

    for i in range(args.number):
        subprocess.run(command)
        os.rename(args.save_directory, f'{args.save_directory}_generation_{i}')
        os.mkdir(args.save_directory)
