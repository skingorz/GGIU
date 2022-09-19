# Gestalt-Guided Image Understanding for Few-Shot Learning
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/yaoyao-liu/mnemonics/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
![last commit](https://img.shields.io/github/last-commit/FrankLuox/FewShotCodeBase)

This repo is based on [LightningFSL](https://github.com/Frankluox/LightningFSL).

To understand the code correctly, it is highly recommended to first quickly go through the [pytorch-lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/), especially [LightningCLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html). It won't be a long journey since pytorch-lightning is built on the top of pytorch.

## :sparkles: News

**[Sep 17, 2022]**
- GGIU is accepted to **ACCV2022**.

### Installation
Just run the command:

```bash
git clone git@github.com:skingorz/GGIU.git
cd GGIU
conda env create -f env.yaml
conda activate baseCode
```



### running an implemented few-shot model

1. Downloading Datasets:
    - miniImageNet

     The data format is as shown in `dataset_and_process/datasets/miniImageNet.py`:
        
2. Training:
    - Choose the corresponding configuration file in `config` (e.g.`set_config_PN_train.py` for PN model), set inside the parameter dataset directory, logging dir as well as other parameters you would like to change.
    - Change `CONFIG_PY` in `train.sh` (e.g., `CONFIG_PY=config/set_config_PN_train.py`).
    - To begin the running, run the command `bash train.sh`

5. Testing:
    - Choose the corresponding configuration file in `config` (e.g. `set_config_PN_test.py` for testing with GGIU), set inside the parameter dataset directory, logging dir, as well as other parameters you would like to change. If add GGIU, choose the configuration named with GGIU, set `is_TTA` to True, and set the value of `lambd`. Otherwise, set `is_TTA` to False. 
    - Change `CONFIG_PY` and `model` in `test.sh` (e.g., `CONFIG_PY=config/set_config_PN_TTA_test.py`  `model=epoch=55-step=13999.ckpt`).
    - To begin the testing, run the command `bash test.sh`
