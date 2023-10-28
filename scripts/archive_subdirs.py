import os
import tarfile
import argparse
from natsort import natsorted
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
args = parser.parse_args()

dir_list = natsorted(os.listdir(args.dir))
for subdir in dir_list:
    current_subdir = os.path.join(args.dir, subdir)
    if os.path.isdir(current_subdir):
        tar_file = current_subdir + ".tar"
        with tarfile.open(tar_file, "w") as tar:
            tar.add(os.path.join(args.dir, subdir), arcname=subdir)
            tar.close()
            shutil.rmtree(current_subdir)
        print(f"Archived {current_subdir} to {tar_file}")
