# BayesBeat
Source code & Pretrained model for our IMWUT (UbiComp) 2022 paper: "[BayesBeat: A Bayesian Deep Learning Approach for Atrial Fibrillation Detection from Noisy Photoplethysmography (PPG) Signals](https://dl.acm.org/doi/10.1145/3517247)" [[preprint](https://arxiv.org/abs/2011.00753)]

The Pretrained pytorch model file for CPU is provided in [`saved_model`](saved_model) folder

## Requirements
```
CUDA version 10.2+
Python version 3.7+
PyTorch version 1.5.1+
```

## How to run:
   - First, setup a virtual environment and activate it
   - Install all the [requirements](requirements.txt) and their dependencies
   - Then download the dataset and put that into proper folder structure
   - Finally, run `python evaluate_bayesian.py`
   - For the version that utilizes gpu to evaluate, please refer to this [repo](https://github.com/Subangkar/BayesBeat): https://github.com/Subangkar/BayesBeat

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

## Citation
If you use our work, please cite:
```bibtex
@article{das2022bayesbeat,
  title={BayesBeat: Reliable Atrial Fibrillation Detection from Noisy Photoplethysmography Data},
  author={Das, Sarkar Snigdha Sarathi and Shanto, Subangkar Karmaker and Rahman, Masum and Islam, Md Saiful and Rahman, Atif Hasan and Masud, Mohammad M and Ali, Mohammed Eunus},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={6},
  number={1},
  pages={1--21},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```
