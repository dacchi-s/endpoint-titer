# Endpoint Titer Analysis Tool

This tool processes ELISA data to calculate antibody titers using optimized logistic regression (4-parameter and 5-parameter logistic models). The tool is designed to be flexible and allows selection of technical replicates and fitting methods for high accuracy.

## Features

- **4-parameter and 5-parameter logistic regression fitting**
- **Evaluation of dilution rates**
- **Calculation of fitting metrics** (e.g., R², Adjusted R², AIC, BIC, RMSE)
- **Plotting of results** and saving to Excel with plots included

## Requirements

- Python 3.8+
- Required packages: `openpyxl`, `pandas`, `numpy`, `matplotlib`, `scipy`

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

For conda users:

```conda environment
conda env create -f endpoint_titer.yml

conda activate endpoint-titer-analysis
```

## Usage

python endpoint-titer.py --input path/to/input.xlsx --cutoff <cutoff-value> --method <4|5|auto> --replicates <1|2> --verbose

## Example

python endpoint-titer.py --input example_data.xlsx --cutoff 0.1 --method auto --replicates 2 --verbose

## Outputs

- Excel file: Results are saved in results_<input_filename>.xlsx.
- Plots: Individual sample plots are saved in a plots folder within the output directory.
