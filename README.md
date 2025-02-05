# Adapting MEDFAIR for analysis of biases in retinal images from the UK Biobank

Work for [paper](https://arxiv.org/abs/2408.02676) to be presented at FAIMI MICCAI 2024 workshop in October.

Forked and adapted from [MEDFAIR](https://github.com/ys-zong/MEDFAIR/blob/main/): fairness benchmarking suite for medical imaging ([paper](https://arxiv.org/abs/2210.01725)). See MEDFAIR documentation [here](https://github.com/ys-zong/MEDFAIR/blob/main/docs/index.md).

## Quick Start

### Installation
Python >= 3.8+ and Pytorch >=1.10 are required for running the code.

Via pip:
```python
cd MEDFAIR/
pip install -r my_requirements.txt
```

Via conda:
```python
cd MEDFAIR/
conda env create -n fair_benchmark -f environment.yml
conda activate fair_benchmark
```

### Dataset
Due to data use agreements, the UKBB retinal images cannot be shared. For those with access, we use R eye images from Datafield 21015. The code could be easily adapted for other retinal imaging or medical imaging datasets.

### Data Preprocessing
See `mynotebooks/UKBB Preprocessing.ipynb` for information on preprocessing of relevant sensitive attributes, splitting into train/val/test sets, and pickling images.

After preprocessing, specify the paths of the metadata and pickle files in `configs/datasets.json`.

### Run a single experiment
```python
python main.py --experiment [experiment] --experiment_name [experiment_name] --dataset_name [dataset_name] \
     --backbone [backbone] --total_epochs [total_epochs] --sensitive_name [sensitive_name] \
     --batch_size [batch_size] --lr [lr] --sens_classes [sens_classes]  --val_strategy [val_strategy] \
     --output_dim [output_dim] --num_classes [num_classes]
```

To reproduce experiments in the paper (replace experiment, sensitive_name, and sens_classes accordingly):

```python
python main.py --experiment baseline --wandb_name [wandb_name] --data_folder [data_folder] --early_stopping 10 --class_name adj_bp --dataset_name UKBB_RET --pretrained True --total_epochs 100 --sensitive_name Centre --batch_size 512 --sens_classes 6 --output_dim 1 --num_classes 1 --random_seed 42 --backbone InceptionV3 --lr 0.0005
```

See `parse_args.py` for more options.

### Analyse results
See `mynotebooks/UKBB Baseline Model Analysis` for an example analysis of the baseline (ERM) model, its disparities, and characterisation of its disparities (eg analysis of model feature representations etc.).

See `mynotebooks/UKBB Mitigation Models Analysis` for an example analysis of the mitigation models, and comparison of their performance.

## Citation
Please consider citing our paper if you find this repo useful.
```
@InProceedings{alloula2024biasesukbiobankbasedretinal,
author="Alloula, Anissa
and Mustafa, Rima
and McGowan, Daniel R.
and Papie{\.{z}}, Bart{\l}omiej W.",
editor="Puyol-Ant{\'o}n, Esther
and Zamzmi, Ghada
and Feragen, Aasa
and King, Andrew P.
and Cheplygina, Veronika
and Ganz-Benjaminsen, Melanie
and Ferrante, Enzo
and Glocker, Ben
and Petersen, Eike
and Baxter, John S. H.
and Rekik, Islem
and Eagleson, Roy",
title="On Biases in a UK Biobank-Based Retinal Image Classification Model",
booktitle="Ethics and Fairness in Medical Imaging",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="140--150"
}
```
## Acknowledgement

We thank MEDFAIR authors and their detailed repo which has formed the basis of this work.
```
@inproceedings{zong2023medfair,
    title={MEDFAIR: Benchmarking Fairness for Medical Imaging},
    author={Yongshuo Zong and Yongxin Yang and Timothy Hospedales},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2023},
}
```
