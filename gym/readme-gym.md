
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
python experiment.py --env hopper --dataset medium --model_type dt
```

Adding `-w True` will log results to Weights and Biases.

## My example
```python
python experiment.py --env halfcheetah --dataset expert --model_type dt -w True

nohup python -u experiment.py --env halfcheetah --dataset expert --model_type de --max_iters 30 -it cat -w True  > de_halfcheetah_expert_30iter_cat_w.log 2>&1 &   ## cost 24h to 45 iters

nohup python -u experiment.py --env halfcheetah --dataset expert --model_type de --max_iters 30 -it seq -w True > de_halfcheetah_expert_30iter_seq_w.log 2>&1 &
```
