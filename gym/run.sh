# Big Model, 3.5-4h for one iter
## total paras: 5330176, but 11G GPU memory
# nohup python -u experiment.py -it cat --gpu_id 0 --dataset CheetahWorld-v2 --env_name cheetah-dir --env_level normal --model_type de --n_layer 5 --n_head 1 --max_iters 30 --batch_size 512 --K 200 -w True  > 0_cat_de_cheetah-dir_normal_30iter_512bs_200K_w.log 2>&1 & 

# ## total paras: 1123072, but 22G GPU memory
# nohup python -u experiment.py -it seq --gpu_id 1 --dataset CheetahWorld-v2 --env_name cheetah-dir --env_level normal --model_type de --n_layer 5 --n_head 1 --max_iters 30 --batch_size 512 --K 200 -w True  > 1_seq_de_cheetah-dir_normal_30iter_512bs_200K_w.log 2>&1 & 

# # env_name='cheetah-dir', env_level='all'
# nohup python -u experiment.py --gpu_id 0 -it cat --dataset CheetahWorld-v2 --env_name cheetah-dir --env_level all --model_type de --n_layer 5 --n_head 1 --max_iters 3 --batch_size 64 --K 200 -w True  > 0_cat_de_cheetah-dir_all_3iter_64bs_200K_w.log 2>&1 & 
# nohup python -u experiment.py --gpu_id 1 -it seq --dataset CheetahWorld-v2 --env_name cheetah-dir --env_level all --model_type de --n_layer 5 --n_head 1 --max_iters 3 --batch_size 64 --K 200 -w True  > 1_seq_de_cheetah-dir_all_3iter_64bs_200K_w.log 2>&1 & 

# # exp_new.py env_name='all', env_level='all', batch_size=64
# nohup python -u exp_new.py --gpu_id 0 -it cat --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --n_layer 5 --n_head 1 --max_iters 3 --batch_size 64 --K 200 -w True  > 0_cat_de_all_all_3iter_64bs_200K_w.log 2>&1 & 
# nohup python -u exp_new.py --gpu_id 1 -it seq --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --n_layer 5 --n_head 1 --max_iters 3 --batch_size 64 --K 200 -w True  > 1_seq_de_all_all_3iter_64bs_200K_w.log 2>&1 & 

# # exp_new.py env_name='all', env_level='all', batch_size=512
# nohup python -u exp_new.py --gpu_id 0 -it cat --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --n_layer 5 --n_head 1 --max_iters 20 --batch_size 512 --K 200 -w True  > 0_cat_de_all_all_20iter_512bs_200K_w.log 2>&1 & 
# nohup python -u exp_new.py --gpu_id 1 -it seq --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --n_layer 5 --n_head 1 --max_iters 20 --batch_size 512 --K 200 -w True  > 1_seq_de_all_all_20iter_512bs_200K_w.log 2>&1 & 


# # exp_new.py env_name='all', env_level='all', batch_size=512, num_steps_per_iter=200
# nohup python -u exp_new.py --gpu_id 0 -it cat --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --n_layer 5 --n_head 1 --max_iters 10 --num_steps_per_iter 200 --batch_size 512 --K 200 -w True  > 0_cat_de_all_all_10iter_200ns_512bs_200K_w.log 2>&1 & 
# nohup python -u exp_new.py --gpu_id 1 -it seq --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --n_layer 5 --n_head 1 --max_iters 10 --num_steps_per_iter 200 --batch_size 512 --K 200 -w True  > 1_seq_de_all_all_10iter_200ns_512bs_200K_w.log 2>&1 & 

# # exp_new.py env_name='all', env_level='all', --embed_dim=256, batch_size=512
# nohup python -u exp_new.py --gpu_id 0 -it cat --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --embed_dim 256 --n_layer 5 --n_head 4 --max_iters 20 -ns 2900 -bs 64 -tl 200 -w True  > de_all_all_256h5layer4head_20iter_2900ns64bs200tl_0_cat_w.log 2>&1 & 
# nohup python -u exp_new.py --gpu_id 1 -it seq --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --embed_dim 256 --n_layer 5 --n_head 4 --max_iters 20 -ns 2900 -bs 64 -tl 200 -w True  > de_all_all_256h5layer4head_20iter_2900ns64bs200tl_1_seq_w.log 2>&1 & 

nohup python -u exp_new.py --gpu_id 0 -it cat --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --embed_dim 160 --n_layer 6 --n_head 4 -bs 512 -K 200 -w True  > de_all_all_160h6layer4head_512bs_0_cat_w.log 2>&1 & 
nohup python -u exp_new.py --gpu_id 1 -it seq --dataset CheetahWorld-v2 --env_name all --env_level all --model_type de --embed_dim 160 --n_layer 6 --n_head 4 -bs 256 -K 200 -w True  > de_all_all_160h6layer4head_256bs_1_seq_w.log 2>&1 & 
