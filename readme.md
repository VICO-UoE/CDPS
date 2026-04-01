# CDPS

Pytorch Implementation of Class-Discriminative Patch Sets [CDPS]

## Requirements
### Environment
Create a conda environment and install the dependencies:
```
conda create -n cdps python=3.9
conda activate cdps
pip install -r requirements.txt
```

### Dataset
Follow [Dataset.md](https://github.com/VICO-UoE/CDPS/blob/main/datasets/dataset.md) to download the 12 datasets.

## Numerical Results
See [Results](https://github.com/VICO-UoE/CDPS/blob/main/datasets/Results.xlsx) for detailed numbers.

## Get Started
### Get fewshot and test samples [Optional]
We provided the fewshot and test samples in `meta_data`.
If you want to create your own samples, run the following code to generate txt files:
```
python generate_fewshots.py --dataset eurosat
```

### Extract features
Run the following code to extract the patch and global features for fewshot and test samples:
```
python extract_features_masks.py --dataset eurosat
```

### Perform the classification
Run the following code to perform the classification of CDPS:
```
python cdps_classification.py --dataset eurosat
```
