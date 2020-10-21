# BayesBeat
Source code & Pretrained model of BayesBeat: A Bayesian Deep Learning Approach for Atrial Fibrillation Detection from Noisy Photoplethysmography (PPG) Signals  
The Pretrained pytorch model file for CPU is provided in [`saved_model`](saved_model) folder

## Requirements
```
CUDA version 10.2
Python version 3.7
PyTorch version 1.5.1
```

## How to run:
   - First, setup a virtual environment and activate it
   - Install all the [requirements](requirements.txt) and their dependencies
   - Then download the dataset and put that into proper folder structure
   - Finally, run `python evaluate_bayesian.py`

**Data Folder Structure for running [`evaluate_bayesian.py`](evaluate_bayesian.py):**
```
data/
    test/
        signal.npy
        qa_label.npy
        rhythm.npy
```

## Additional Files:
[distr_split_ids.npy](distr_split_ids.npy): A dictionary that contains list of individal ids for train, validation & test set for the distribution of dataset
