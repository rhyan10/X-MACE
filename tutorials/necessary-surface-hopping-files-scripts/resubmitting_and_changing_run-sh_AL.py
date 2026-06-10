import os
import subprocess

# Define the root directory containing subdirectories
root_directory = os.getcwd()

# Iterate through each folder in the root directory
for folder in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder)

    #iterate through subfolders in folders Singlet_1 and Singlet_2
    if folder == "Singlet_1" or folder == "Singlet_2":
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            # Check if it's a directoryi
            output_file = os.path.join(subfolder_path, "output.lis")
            if os.path.isdir(subfolder_path) and not os.path.exists(output_file):
                slurm_script = f"""#!/bin/bash
#SBATCH --job-name={subfolder}-SH
#SBATCH --output={subfolder_path}/slurm_%j.out
#SBATCH --time=20:00:00
#SBATCH --signal=TERM@300   # send SIGTERM 5 minutes before timeout
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=14G
#SBATCH --partition=YOURPARTITION

WORK_DIR={subfolder_path}
cd {subfolder_path}
cp 

source ~/.bashrc

$SHARC/driver.py -i mace input &> driver.log
"""
                slurm_script_path = os.path.join(subfolder_path, "run.sh")
                # Write the Slurm script to the directory
                with open(slurm_script_path, "w") as script_file:
                    script_file.write(slurm_script)
                subprocess.run(["sbatch", slurm_script_path])
                print(f"{subfolder} submitted")

        else:
                print(f"{subfolder} is not a directory. Skipping.")
