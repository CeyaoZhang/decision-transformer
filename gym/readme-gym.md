
# OpenAI Gym

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

## Example usage

Experiments can be reproduced with the following:

```
python experiment.py --env_name hopper --env_level medium --model_type dt
```

Adding `-w True` will log results to Weights and Biases.

# My example

## use the D4RL dataset
```python
python experiment.py --env_name halfcheetah --env_level expert --model_type dt -w True

## cost 24h to 45 iters, and 17h for 30 iters
nohup python -u experiment.py --env_name halfcheetah --env_level expert --model_type de --max_iters 30 -it cat -w True  > de_halfcheetah_expert_30iter_cat_w.log 2>&1 &   

nohup python -u experiment.py --env_name halfcheetah --env_level expert --model_type de --max_iters 30 -it seq -w True > de_halfcheetah_expert_30iter_seq_w.log 2>&1 &
```

## use my CheetahWorld-v2 data

You can download the dataset from [CheetahWorld-v2](https://drive.google.com/drive/folders/1g0u7dFNf0lSC8K66yNNR0aprM0yBDrT6?usp=share_link).


## Use One GPU 

### Try single gpu version
```
python exp_single_gpu.py --gpu_id 0 -it cat --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --embed_dim 160 --n_layer 6 --n_head 4 -bs 512 -K 200 -w True
```

### Try multi gpu version with world_size=1
```
python exp_multi_gpu.py --world_size 1 -it cat --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --embed_dim 160 --n_layer 6 --n_head 4 -bs 512 -K 200 -w True
```

## Use Multi GPU 

### Try multi gpu version with world_size
```
python exp_multi_gpu.py --world_size 2 -it cat --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --embed_dim 160 --n_layer 6 --n_head 4 -bs 512 -K 200 -w True
```