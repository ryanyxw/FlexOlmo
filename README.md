<div align="center">
  <!-- <img src="https://github.com/allenai/OLMo/assets/8812459/774ac485-a535-4768-8f7c-db7be20f5cc3" width="300"/> -->
  <img src="https://github.com/allenai/FlexOlmo/blob/main/assets/FlexOlmo_Logo.png" alt="FlexOlmo Logo" width="280" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <h1>FlexOlmo</h1>
</div>
<p align="center">
  <a href="https://github.com/allenai/FlexOlmo/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://discord.gg/sZq3jTNVNG">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=Ai2&color=%235B65E9">
  </a>
</p>

FlexOlmo is a new kind of LM that unlocks a new paradigm of data collaboration. With FlexOlmo, data owners can contribute to the development of open language models without giving up control of their data. There is no need to share raw data directly, and data contributors can decide when their data is active in the model, deactivate it at any time, and receive attributions whenever it's used for inference.


## Installation

We recommend using Python 3.10. First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system and hardware. 

To install dependencies, run:

```bash
git clone https://github.com/allenai/FlexOlmo.git
cd FlexOlmo
conda create -n flexolmo python=3.10
conda activate flexolmo
pip install -e ".[train,beaker,wandb]"  # for training
```

FlexOlmo is built using [OLMo-core](https://github.com/allenai/OLMo-core.git). OLMo-core's published [Docker images](https://github.com/orgs/allenai/packages?repo_name=OLMo-core) contain all core and optional dependencies. You can also adapt our [Dockerfile](https://github.com/allenai/FlexOlmo/blob/main/src/Dockerfile) to build your own images.

## Model Summary

[FlexOlmo-7x7B-1T](https://huggingface.co/allenai/FlexOlmo-7x7B-1T) (without router training) is a Mixture-of-Experts model with 33B total parameters, combining independently trained experts on public-mix, news, math, code, academic texts, creative writing, and Reddit data. The public-mix expert is trained on 1T tokens of public data while the other experts are branched from the public-mix expert and trained on 50B tokens of their respective data.

| Model | Description |
|-------|--------|
| [Flex-news-2x7B-1T](https://huggingface.co/allenai/Flex-news-2x7B-1T) | News |
| [Flex-math-2x7B-1T](https://huggingface.co/allenai/Flex-math-2x7B-1T) | Math |
| [Flex-code-2x7B-1T](https://huggingface.co/allenai/Flex-code-2x7B-1T) | Code |
| [Flex-pes2o-2x7B-1T](https://huggingface.co/allenai/Flex-pes2o-2x7B-1T) | Academic |
| [Flex-creative-2x7B-1T](https://huggingface.co/allenai/Flex-creative-2x7B-1T) | Creative Writing |
| [Flex-reddit-2x7B-1T](https://huggingface.co/allenai/Flex-reddit-2x7B-1T) | Reddit |
| [FlexOlmo-7x7B-1T](https://huggingface.co/allenai/FlexOlmo-7x7B-1T) | Combined FlexOlmo model | 
| [FlexOlmo-7x7B-1T-RT](https://huggingface.co/allenai/FlexOlmo-7x7B-1T) | Combined FlexOlmo model with optional router training on proxy mix| 


## Training scripts

All python training scripts can be found in [src/scripts/train](src/scripts/train/). These scripts are meant to be launched with `torchrun`.
The scripts can also be [launched with beaker](src/scripts/beaker/).


### Evaluation

| **Model** | **MC9** | **Gen5** | **MMLU** | **MMLU Pro** | **AGIEval** | **BBH** | **Math2** | **NewsG** | **PoemG** | **SciRIFF5** | **Code4** | **Avg.** |
|----------|--------|----------|----------|--------------|-------------|---------|-----------|-----------|-----------|--------------|-----------|----------|
| Prev. Public model | 68.7 | 58.8 | 55.9 | 26.2 | 39.9 | 35.7 | 8.2 | 76.0 | 47.8 | 48.1 | 1.1 | 42.4 |
| **Individual** |
| [Math](https://huggingface.co/allenai/Flex-math-2x7B-1T) | 62.5 | 44.3 | 50.6 | 24.1 | 42.0 | 45.6 | **53.1** | 42.6 | 28.0 | 50.7 | 15.8 | 41.8 |
| [Code](https://huggingface.co/allenai/Flex-code-2x7B-1T) | 40.5 | 39.4 | 29.5 | 14.5 | 27.4 | 38.1 | 6.0 | 45.1 | 28.2 | 48.0 | 21.0 | 30.7 |
| Educational Text | 64.3 | 52.1 | 56.5 | 27.0 | 39.7 | 40.3 | 13.6 | 57.6 | 51.8 | 51.7 | 3.0 | 41.6 |
| [News](https://huggingface.co/allenai/Flex-news-2x7B-1T) | 46.5 | 48.6 | 36.4 | 15.2 | 25.7 | 30.9 | 2.5 | 77.7 | 26.9 | 47.0 | 0.0 | 32.5 |
| [Creative Writing](https://huggingface.co/allenai/Flex-creative-2x7B-1T) | 42.7 | 43.9 | 31.5 | 11.6 | 23.3 | 27.6 | 1.7 | 56.9 | **67.5** | 42.4 | 0.0 | 31.7 |
| [Academic](https://huggingface.co/allenai/Flex-pes2o-2x7B-1T) | 41.0 | 45.2 | 33.8 | 14.8 | 24.1 | 32.4 | 6.5 | 51.8 | 23.0 | 52.0 | 0.0 | 29.5 |
| [Reddit](https://huggingface.co/allenai/Flex-reddit-2x7B-1T) | 64.7 | 36.5 | 56.1 | 25.5 | 35.5 | 19.7 | 2.5 | 54.1 | 8.6 | 32.7 | 1.7 | 30.7 |
| **Combined** |
| BTM (top-2) | 68.7 | 57.7 | 59.4 | 28.3 | 43.2 | 44.3 | 23.1 | 73.6 | 54.4 | 46.3 | **24.0** | 47.6 |
| 🔥 **FlexOlmo-7x7B-1T** | **70.4** | **60.1** | **60.2** | **30.5** | 44.8 | 46.8 | 47.9 | 77.2 | 66.2 | 53.8 | 14.6 | 52.0 |
| **FlexOlmo-7x7B-1T-RT** | 70.3 | 60.0 | **60.2** | 30.3 | **45.2** | **47.2** | 47.7 | **78.3** | **67.6** | **53.9** | 13.3 | **52.2** |


## Citation

Scripts for experiments in the paper can be found in [scripts](scripts/).

Coming soon