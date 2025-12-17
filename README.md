# GAMERS
Repository for General Approach for Macromolecular Equilibration via Restrained Simulations

<br />
<img src="overview.png" />
<br />

This README explains how to run the GAMERS pipeline using the scripts provided in this repository. It assumes you have cloned the repo and will edit a single configuration file before running the pipeline.

## Quick start commands

- **Clone the repository:**
  ```bash
  git clone https://github.com/webbtheosim/GAMERS.git
  cd GAMERS
  ```

- **Edit the configuration:**
  ```bash
  # open GAMERS.input at repository root and adjust:
  output_directory:/absolute/path/to/your/workdir
  ```

- **Run Stage I (in order):**
  ```bash
  python stage_I/step_1.py
  python stage_I/step_2_in_maker.py
  lammps stage_I/step_2/inputs/step_2.in
  python stage_I/step_3.py
  ```

- **Generate or provide input files for Stage II:**
  ```bash
  python openff_system_maker.py
  ```
  This will generate the required files with openff-toolkit, namely
  <ol type="1">
    <li>a .pdb file containing all atoms of the system named sys.pdb</li>
    <li>an openmm system object matching sys.pdb saved as sys.xml</li>
  </ol>
  
- **Run Stage II:**
  ```bash
  python stage_II/stage_II.py
  ```

Outputs will be written to the locations configured in GAMERS.input (see Configuration below).

## File map and purpose
- `GAMERS.input` — central configuration file read by all Python scripts; contains /absolute/path/to/your/workdir and system-specific settings.
- `stage_I/step_1.py` — creates the KG → HR mapping and prepares the LAMMPS data/topology files for Step 2.
- `stage_I/step_2_in_maker.py` — generates step_2.in (LAMMPS input) by adding parameters to step_2_base.in.
- `stage_I/step_3.py` — creates restraint point IDs and positions used by Stage II.
- `stage_II/stage_II.py` — runs the OpenMM-based Stage II simulation that consumes restraint files produced by Stage I.
- `openff_system_maker.py` — optional generation of OpenMM input files (e.g. sys.xml) with SAGE force field (use as needed in your workflow).

**Place-to-file relationships:**
- Files produced by `stage_I/step_1.py` → consumed by `stage_I/step_2_in_maker.py` and `stage_I/step_3.py`.
- `stage_I/step_2_in_maker.py` → produces `stage_I/step_2_inputs/step_2.in` from stage_I/step_2_inputs/step_2_base.in.
- `stage_I/step_3.py` → produces restraint ID/coordinate files consumed by `stage_II/stage_II.py`.
- `stage_II/stage_II.py` → produces final position/velocity files, found in `stage_II/final_positions`, for each phase of GAMERS.

## Configuration and PATH_TO_DIRECTORY
- Edit the single file `GAMERS.input` at the repository root to configure runs.
- Set the variable `/absolute/path/to/your/workdir` to an absolute path for your working directory:
  ```bash
  output_directory:/absolute/path/to/your/workdir
  ```

- All scripts read configuration from `GAMERS.input`. Changing only `GAMERS.input` will redirect where inputs and outputs are read/written.

## Dependencies
- Python 3.8+ (use a virtual environment for reproducibility).
- Typical Python packages used by the scripts (install into the venv): `numpy`, `openmm`, `openff-toolkit`, `rdkit`, and any other packages imported at the top of the scripts.
- LAMMPS — required to execute the generated `step_2.in`.

## Troubleshooting and notes
- If a script fails to find files, confirm `/absolute/path/to/your/workdir` in `GAMERS.input` and ensure that directory exists and is writable.
- If you see import errors, install the missing packages into the active Python environment.
- If you modified scripts to inline variables, revert them to read from `GAMERS.input` to keep a single authoritative configuration.
- Logging and intermediate files: check the directories configured in `GAMERS.input` for outputs created by each step.
- Typical run order must be preserved: `step_1.py` → `step_2_in_maker.py` → `step_2.in` → `step_3.py` → `stage_II.py`.

## License and attribution
- Repository source: https://github.com/webbtheosim/GAMERS
- Please keep attribution to the original GAMERS authors in derivative documentation.
