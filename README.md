# CETERO-S

CETERO-S is a Python tool for the analysis of J–R resistance curves from compact tension C(T) fracture tests using the unloading compliance method.

The software processes experimental load, displacement and CMOD data, detects unloading cycles, estimates compliance values, applies rotation correction for C(T) specimens, calculates crack growth and constructs J–R resistance curves following ASTM E1820-inspired procedures.

---

## Features

- Import of experimental load, displacement and CMOD data
- Automatic detection of unloading cycles
- Generation of multiple compliance candidates
- Supervised compliance selection
- Rotation correction for C(T) specimens
- Crack growth estimation
- Calculation of J-integral components
- Construction of J–R resistance curves
- ASTM E1820-inspired post-processing and validity analysis

---

## Requirements

Python 3.10 or newer is recommended.

Required packages:

- numpy
- pandas
- matplotlib
- scipy

Install them using:

```bash
pip install -r requirements.txt
```

---

## Usage

Run:

```bash
python CETERO-S.py
```

The program will ask the user to select a CSV file containing the experimental data.

---

## Input data

The input CSV file should contain at least the following channels:

| Column      | Description |
|-------------|-------------|
| Channel_0   | Crosshead displacement |
| Channel_1   | Load |
| Channel_2   | CMOD |

These column names can be modified in the code configuration section.

---

## Workflow

The typical analysis workflow is:

1. Import experimental data
2. Detect unloading cycles
3. Generate compliance candidates
4. Select representative compliances
5. Apply rotation correction
6. Estimate crack size and crack growth
7. Calculate J-integral values
8. Construct the J–R curve
9. Perform ASTM E1820-inspired post-processing

---

## Output

During the analysis, CETERO-S generates plots and intermediate results that allow the user to review the complete workflow, including:

- load versus displacement curves
- load versus CMOD curves
- compliance versus CMOD curves
- corrected compliance curves
- crack growth estimation
- J–R resistance curves
- ASTM-inspired post-processing plots

---

## Example

An example dataset is available in the `examples/` folder.

---

## Documentation

Additional documentation and workflow description are available in the `docs/` folder.

---

## Citation

If you use CETERO-S in your work, please cite this repository.

A formal DOI will be provided through Zenodo after the first public release.

---

## License

This project is distributed under the MIT License.

---

## Author

Mechanical Engineer — Universidad de Buenos Aires (UBA)  
M.Sc. in Materials Science and Technology — Instituto Sábato (CNEA–UNSAM)
