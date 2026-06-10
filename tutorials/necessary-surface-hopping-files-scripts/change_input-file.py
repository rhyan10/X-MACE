import os
import re

# Replace this with your top-level directory
top_level_dir = os.getcwd()

# The input template, leaving {state_line} and {rngseed} to be filled in
new_input_template = """printlevel 2

geomfile "geom"
veloc external
velocfile "veloc"
nstates 2 0 0
actstates 2 0 0
{state_line}
coeff auto
rngseed {rngseed}
charge -1 0 0
ezero    {ezero}
tmax 7000.000000
stepsize 0.500000
nsubsteps 25
surf diagonal
coupling ktdc
gradcorrect
ekincorrect parallel_vel
reflect_frustrated none
decoherence_scheme edc
decoherence_param 0.1
hopping_procedure sharc
grad_all
nospinorbit
output_format ascii
output_dat_steps 1
"""

# Walk through the file tree
for root, dirs, files in os.walk(top_level_dir):
    # Only process subdirectories matching Singlet_*
    if os.path.basename(root).startswith("Singlet_"):
        singlet_folder = os.path.basename(root)

        # Determine the correct state line
        if singlet_folder == "Singlet_1":
            state_line = "state 2 mch"
        elif singlet_folder == "Singlet_2":
            state_line = "state 3 mch"
        else:
            # If neither Singlet_1 nor Singlet_2, skip or set a default
            print(f"Warning: Unrecognized folder {singlet_folder}, skipping...")
            continue

        # Now look inside subdirectories of this Singlet_* folder
        for subdir in dirs:
            subsub_path = os.path.join(root, subdir)
            input_file_path = os.path.join(subsub_path, "input")

            if os.path.isfile(input_file_path):
                with open(input_file_path, "r") as file:
                    content = file.read()

                # Extract rngseed (support positive or negative)
                match = re.search(r"rngseed\s+([-+]?\d+)", content)
                rngseed = match.group(1) if match else "24999"  # Default

                match = re.search(r"ezero\s+([-+]?\d*\.?\d+)", content)
                if match:
                    ezero = match.group(1)

                # Create the new input content
                new_content = new_input_template.format(
                    state_line=state_line,
                    rngseed=rngseed,
                    ezero=ezero
                )

                # Overwrite the input file
                with open(input_file_path, "w") as file:
                    file.write(new_content)

                print(f"Updated: {input_file_path}")

