# Import the subprocess module
import subprocess

# Define the path to Blender executable
blender_path = "C:\\Program Files\\Blender Foundation\\Blender 2.93\\blender.exe"

# Define the path to the script that opens Blender, deletes all objects, creates a plane and merges it into a single vertex
script_path = "C:\\Users\\User\\Documents\\blender_script.py"

# Run Blender with the subprocess module
subprocess.run([blender_path, "-b", "-P", script_path])
