import regression
import argparse
import pandas as pd
import json
import numpy as np
import os
import pdb

os.chdir("/home/renzo/workspace/epistatic_prior/nn4dms")

model1 = argparse.Namespace(batch_size=64, cluster='local', compress_everything=False, 
                          dataset_file='', dataset_name='temp', delete_checkpoints=False, 
                          early_stopping=True, early_stopping_allowance=10, encoding='one_hot,aa_index', 
                          epoch_checkpoint_interval=1, epoch_evaluation_interval=5, epochs=150, graph_fn='', 
                          learning_rate=0.0001, log_dir_base='output/proteingym_evaluation', min_loss_decrease=1e-05, 
                          net_file='network_specs/cnns/cnn-5xk3f128.yml', np_rseed=7, process='local', py_rseed=7, 
                          split_dir='', split_rseed=7, step_display_interval=0.1, test_size=0.2, tf_rseed=7, 
                          train_size=0.6, tune_size=0.2, wt_aa='', wt_ofs='')

model2 = argparse.Namespace(batch_size=64, cluster='local', compress_everything=False, 
                          dataset_file='', dataset_name='yap1', delete_checkpoints=False, 
                          early_stopping=True, early_stopping_allowance=10, encoding='one_hot,aa_index', 
                          epoch_checkpoint_interval=1, epoch_evaluation_interval=5, epochs=150, graph_fn='', 
                          learning_rate=0.0001, log_dir_base='output/proteingym_evaluation', min_loss_decrease=1e-05, 
                          net_file='network_specs/cnns/cnn-1xk17f32.yml', np_rseed=7, process='local', py_rseed=7, 
                          split_dir='', split_rseed=7, step_display_interval=0.1, test_size=0.2, tf_rseed=7, 
                          train_size=0.6, tune_size=0.2, wt_aa='', wt_ofs='')

model3 = argparse.Namespace(batch_size=32, cluster='local', compress_everything=False, 
                          dataset_file='', dataset_name='yap1', delete_checkpoints=False, 
                          early_stopping=True, early_stopping_allowance=10, encoding='one_hot,aa_index', 
                          epoch_checkpoint_interval=1, epoch_evaluation_interval=5, epochs=150, graph_fn='', 
                          learning_rate=0.0001, log_dir_base='output/proteingym_evaluation', min_loss_decrease=1e-05, 
                          net_file='network_specs/cnns/cnn-3xk17f128.yml', np_rseed=7, process='local', py_rseed=7, 
                          split_dir='', split_rseed=7, step_display_interval=0.1, test_size=0.2, tf_rseed=7, 
                          train_size=0.6, tune_size=0.2, wt_aa='', wt_ofs='')

protein_gym = pd.read_csv("/home/renzo/workspace/datasets/protein_gym.csv")
mult_mutations = protein_gym[protein_gym['includes_multiple_mutants']== True]
sizes = [10, 100,  464, 1000, 4641]
for id in np.array(['yap1']):
    id = id.split('_')[0].lower()
    model1.dataset_name=id
    model2.dataset_name=id
    model3.dataset_name=id
    dir = f"/home/renzo/workspace/epistatic_prior/nn4dms/data/{id}/splits"
    splits = [name for name in os.listdir(dir)]
    splits.sort(key = lambda x : x.split('_')[-1])
    for split in splits:
        type = "_".join(split.split("_")[:-1])
        n = int(split.split("_")[-1].split("-")[0])
        if n not in sizes:
            continue
        print(split)
        model1.log_dir_base=f'output/proteingym_evaluation/{id}/{type}/{split}'
        model2.log_dir_base=f'output/proteingym_evaluation/{id}/{type}/{split}'
        model3.log_dir_base=f'output/proteingym_evaluation/{id}/{type}/{split}'
        split = os.path.join(dir, split)
        model1.split_dir = split
        model2.split_dir = split
        model3.split_dir = split
        regression.main(model3)
        regression.main(model2)
        regression.main(model1)

