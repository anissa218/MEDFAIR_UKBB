# Adapting MEDFAIR for analysis of biases in retinal images from the UK Biobank


Forked and adapted from [MEDFAIR](https://github.com/ys-zong/MEDFAIR/blob/main/): fairness benchmarking suite for medical imaging ([paper](https://arxiv.org/abs/2210.01725)). 

See MEDFAIR documentation [here](https://github.com/ys-zong/MEDFAIR/blob/main/docs/index.md).

## Quick Start

### Installation
Python >= 3.8+ and Pytorch >=1.10 are required for running the code. Other necessary packages are listed in [`environment.yml`](../environment.yml).

### Installation via conda:
```python
cd MEDFAIR/
conda env create -n fair_benchmark -f environment.yml
conda activate fair_benchmark
```

### Dataset
Due to the data use agreements, we cannot directly share the download link. Please register and download datasets using the links from the table below:


### Data Preprocessing
See `notebooks/HAM10000.ipynb` for an simple example of how to preprocess the data into desired format. You can also find other preprocessing scripts for corresponding datasets.
Basically, it contains 3 steps:
1. Preprocess metadata.
2. Split to train/val/test set
3. Save images into pickle files (optional -- we usually do this for 2D images instead of 3D images, as data IO is not the bottleneck for training 3D images).

After preprocessing, specify the paths of the metadata and pickle files in `configs/datasets.json`.


### Run a single experiment
```python
python main.py --experiment [experiment] --experiment_name [experiment_name] --dataset_name [dataset_name] \
     --backbone [backbone] --total_epochs [total_epochs] --sensitive_name [sensitive_name] \
     --batch_size [batch_size] --lr [lr] --sens_classes [sens_classes]  --val_strategy [val_strategy] \
     --output_dim [output_dim] --num_classes [num_classes]
```

For example, for running `ERM` in `HAM10000` dataset with `Sex` as the sensitive attribute:
```python
python main.py --experiment baseline --dataset_name HAM10000 \
     --total_epochs 20 --sensitive_name Sex --batch_size 1024 \
     --sens_classes 2 --output_dim 1 --num_classes 1
```

See `parse_args.py` for more options.

### Run a grid search on a Slurm cluster/Regular Machine
```python
python sweep/train-sweep/sweep_batch.py --is_slurm True/False
```
Set the other arguments as needed.

### Model selection and Results analysis
See `notebooks/results_analysis.ipynb` for a step by step example.

## Citation
Please consider citing our paper if you find this repo useful.
```
@inproceedings{zong2023medfair,
    title={MEDFAIR: Benchmarking Fairness for Medical Imaging},
    author={Yongshuo Zong and Yongxin Yang and Timothy Hospedales},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2023},
}
```

## Acknowledgement
MEDFAIR adapts implementations from many repos (check [here](docs/reference.md#debiasing-methods) for the original implementation of the algorithms), as well as many other codes. Many thanks!
