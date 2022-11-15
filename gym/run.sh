# Big Model

nohup python -u experiment.py --dataset CheetahWorld-v2 --env_name cheetah-dir --env_level normal --model_type de --n_layer 5 --n_head 5 --max_iters 30 --K 200 -it cat -w True  > de_cheetah-dir_normal_30iter_cat_w.log 2>&1 & 
nohup python -u experiment.py --dataset CheetahWorld-v2 --env_name cheetah-dir --env_level normal --model_type de --n_layer 5 --n_head 5 --max_iters 30 --K 200 -it seq -w True  > de_cheetah-dir_normal_30iter_seq_w.log 2>&1 & 