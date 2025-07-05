# FlexOlmo

FlexOlmo is a new kind of LM that unlocks a new paradigm of data collaboration. With FlexOlmo, data owners can contribute to the development of open language models without giving up control of their data. There is no need to share raw data directly, and data contributors can decide when their data is active in the model, deactivate it at any time, and receive attributions whenever it's used for inference.

## Getting started

### Installation

```bash
cd FlexOlmo
conda create -n flexolmo python=3.10
conda activate flexolmo
pip install -e ".[train,beaker,wandb,dev]"  # for training
```

### Training scripts

All training scripts can be found in [src/scripts/train](src/scripts/train/). These scripts are meant to be launched with `torchrun`.


### Train an expert model

2x7B expert

### Merge experts

common merging script

### Evaluate models

Run eval script by specifying model checkpoint

## Results

Add table from HF model card

## Citation

TBD