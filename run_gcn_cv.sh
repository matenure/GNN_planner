##!/usr/bin/env bash
source activate conda_python3
python -u gcn_cv_samegate.py --data_dir=./10split_data --save_dir=./ --data_type=lifted --index_type=random --random_seed=123 --learning_rate=0.001 --hidden_size=100 --num_timesteps=4
#python -u gcn_cv_samegate.py --data_dir=/dccstor/graph_tengfei/masterplan_cv --save_dir=./saved --data_type=lifted --index_type=random --random_seed=123 --learning_rate=0.001 --hidden_size=100 --num_timesteps=4