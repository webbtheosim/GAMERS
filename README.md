# GAMERS
Repository for General Approach for Macromolecular Equilibration via Restrained Simulations

<br />
<img src="overview.png" />
<br />

This README explains how to run the GAMERS pipeline using the scripts provided in this repository. It assumes you have cloned the repo and will edit a single configuration file before running the pipeline.

Quick start commands
- Clone the repository:
git clone https://github.com/webbtheosim/GAMERS.git
cd GAMERS


- Edit the configuration:
open GAMERS.input at repository root and set:
PATH_TO_DIRECTORY = /absolute/path/to/your/workdir


- Run Stage I (in order):
python stage_I/step_1.py
python stage_I/step_2_in_maker.py
python stage_I/step_3.py


- Run Stage II:
python stage_II/openmm.py


Outputs will be written to the locations configured in GAMERS.input (see Configuration below).

File map and purpose
- GAMERS.input — central configuration file read by all Python scripts; contains variables such as PATH_TO_DIRECTORY and other system-specific settings.
- stage_I/step_1.py — creates the KG → HR mapping and prepares the LAMMPS data/topology files for Step 2.
- stage_I/step_2_in_maker.py — generates step_2.in (LAMMPS input).
- stage_I/step_3.py — creates restraint point IDs and positions used by Stage II.
- stage_II/openmm.py — runs the OpenMM-based Stage II simulation that consumes restraint files produced by Stage I.
- openff_system_generation/ — optional helpers for generating starting systems (use as needed in your workflow).
Place-to-file relationships:
- Files produced by step_1.py → consumed by step_2_in_maker.py.
- step_2_in_maker.py → produces step_2.in and LAMMPS data files.
- step_3.py → produces restraint ID/coordinate files consumed by stage_II/openmm.py.

Configuration and PATH_TO_DIRECTORY
- Edit the single file GAMERS.input at the repository root to configure runs.
- Set the variable PATH_TO_DIRECTORY to an absolute path for your working directory:
PATH_TO_DIRECTORY = /absolute/path/to/your/workdir


- All scripts read configuration from GAMERS.input. Changing only PATH_TO_DIRECTORY will redirect where inputs and outputs are read/written.
Confirmation about changing only the path:
- If every Python script in your working copy reads variables from GAMERS.input (as in the repository), updating PATH_TO_DIRECTORY is the only required change to point the pipeline at your filesystem locations. With that change in place, the pipeline should run without further edits.

Dependencies and recommended environment
- Python 3.8+ (use a virtual environment for reproducibility).
- Typical Python packages used by the scripts (install into the venv): numpy, pandas and any other packages imported at the top of the scripts. Create a venv and install via:
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install numpy pandas


- LAMMPS — required if you plan to execute the generated step_2.in.
- OpenMM — required to run stage_II/openmm.py.
If you want, generate a requirements.txt by listing the imports found at the top of each script and running pip freeze inside a working environment.

Troubleshooting and notes
- If a script fails to find files, confirm PATH_TO_DIRECTORY in GAMERS.input and ensure that directory exists and is writable.
- If you see import errors, install the missing packages into the active Python environment.
- If you modified scripts to inline variables, revert them to read from GAMERS.input to keep a single authoritative configuration.
- Logging and intermediate files: check the directories configured in GAMERS.input for outputs created by each step.
- Typical run order must be preserved: step_1.py → step_2_in_maker.py → step_3.py → openmm.py.

License and attribution
- Repository source: https://github.com/webbtheosim/GAMERS/tree/main
- Please keep attribution to the original GAMERS authors in derivative documentation.
