<p align="center">
  <h1 align="center">Training-free Discriminative Patch Mining for Robust Few-Shot Recognition with CLIP
</h1>
  <p align="center">
    <a href="https://zhenzhang-ye.github.io/">Zhenzhang Ye</a>
    ·
    <a href="https://danier97.github.io/">Duolikun Danier</a>
    ·
    <a href="https://www.bozhao.me/">Bo Zhao</a>
    ·
    <a href="https://homepages.inf.ed.ac.uk/hbilen/">Hakan Bilen</a>
  </p>
  <p align="center">
    ECCV 2026
  </p>

  PyTorch implementation of Training-free Discriminative Patch Mining for Robust Few-Shot Recognition with CLIP [[Paper](https://groups.inf.ed.ac.uk/vico/assets/pdf/Ye26.pdf)].

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
