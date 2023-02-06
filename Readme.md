
# Benchmark scripts for Text OOD detection 

This is the sister repository of [https://github.com/icannos/Todd/](https://github.com/icannos/Todd/). 
This repository contains the scripts and dataset loading for the Todd benchmark. It provides easy access to a range
of datasets and models on which to run Todd detectors.

## Cite this work

If you use this work, please cite us using:

```
@software{Darrin_Todd_A_tool_2023,
author = {Darrin, Maxime and Faysse, Manuel and Staerman, Guillaume and Picot, Marine and Dadalto Camara Gomez, Eduardo and Colombo, Pierre},
month = {2},
title = {{Todd: A tool for text OOD detection.}},
url = {https://github.com/icannos/Todd},
version = {0.0.1},
year = {2023}
}
```

## Installation

To install the package, run the following command:

```bash
git clone git@github.com:icannos/ToddBenchmark.git
cd ToddBenchmark
pip install -e .
```

## Available datasets

Please see `toddbenchmark/classification_datasets_config.py` and
`toddbenchmark/generation_datasets_configs.py` for the list of available datasets and configuration.

## How to add a new dataset

To add a new dataset you'll need 
- To add corresponding configs in `toddbenchmark/(*)_datasets_config.py` and
- To add a load function in `toddbenchmark/(*)_datasets.py`
- To add the loading condition in the corresponding "load_requested_dataset" / "prep_dataset" function in `toddbenchmark/(*)_datasets.py`
