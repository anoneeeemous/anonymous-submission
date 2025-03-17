# Reproducing Experiments

This directory contains code to reproduce the experiments from our paper on ProuDT. For GradTree implementation, you can refer to the source code here: [GradTree Repository](https://github.com/s-marton/GradTree).

## Overview

The `experiments/experiment.py` script allows you to run the same experiments presented in our paper. It supports:
- Running on all datasets or a specific dataset
- Configuring tree depth
- Controlling the number of experimental trials 


## Usage

```bash
python experiments/experiment.py [--datasetname DATASET] [--depth DEPTH] [--trials TRIALS]
```

### Parameters

- `--datasetname`: (Optional) Name of a specific dataset to run. If not provided, all datasets will be run sequentially.
- `--depth`: (Optional) Tree depth to use. Default is 8 for small datasets, 11 for large datasets.
- `--trials`: (Optional) Number of trials to run. Default is 100 for small datasets, 10 for large datasets.

### Examples

Run on all datasets to reproduce full experiments:
```bash
python experiments/experiment.py
```

Run on a specific small dataset with default parameters (will run 100 trials):
```bash
python experiments/experiment.py --datasetname car
```

Run on pendigits with a custom depth and fewer trials:
```bash
python experiments/experiment.py --datasetname pendigits --depth 9 --trials 5
```



## Requirements

- All dependencies from the main project
- Datasets should be available in the /datasets directory. Please unzip the file.
- For large datasets, sufficient RAM and preferably a GPU




