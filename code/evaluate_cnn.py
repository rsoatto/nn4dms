import regression
import argparse
import pandas as pd
import json
import numpy as np
import os
import pdb
from tqdm import tqdm

os.chdir("/home/renzo/workspace/epistatic_prior/nn4dms")

model1 = argparse.Namespace(batch_size=64, cluster='local', compress_everything=False, 
                          dataset_file='', dataset_name='temp', delete_checkpoints=True, 
                          early_stopping=False, early_stopping_allowance=10, encoding='one_hot,aa_index', 
                          epoch_checkpoint_interval=1, epoch_evaluation_interval=5, epochs=150, graph_fn='', 
                          learning_rate=0.0001, log_dir_base='output/proteingym_evaluation', min_loss_decrease=1e-05, 
                          net_file='network_specs/cnns/cnn-5xk3f128.yml', np_rseed=7, process='local', py_rseed=7, 
                          split_dir='', split_rseed=7, step_display_interval=0.1, test_size=0.2, tf_rseed=7, 
                          train_size=0.6, tune_size=0.2, wt_aa='', wt_ofs='')

model2 = argparse.Namespace(batch_size=64, cluster='local', compress_everything=False, 
                          dataset_file='', dataset_name='temp', delete_checkpoints=True, 
                          early_stopping=False, early_stopping_allowance=10, encoding='one_hot,aa_index', 
                          epoch_checkpoint_interval=1, epoch_evaluation_interval=5, epochs=150, graph_fn='', 
                          learning_rate=0.0001, log_dir_base='output/proteingym_evaluation', min_loss_decrease=1e-05, 
                          net_file='network_specs/cnns/cnn-1xk17f32.yml', np_rseed=7, process='local', py_rseed=7, 
                          split_dir='', split_rseed=7, step_display_interval=0.1, test_size=0.2, tf_rseed=7, 
                          train_size=0.6, tune_size=0.2, wt_aa='', wt_ofs='')

model3 = argparse.Namespace(batch_size=32, cluster='local', compress_everything=False, 
                          dataset_file='', dataset_name='temp', delete_checkpoints=True, 
                          early_stopping=False, early_stopping_allowance=10, encoding='one_hot,aa_index', 
                          epoch_checkpoint_interval=1, epoch_evaluation_interval=5, epochs=27, graph_fn='', 
                          learning_rate=0.0001, log_dir_base='output/proteingym_evaluation', min_loss_decrease=1e-05, 
                          net_file='network_specs/cnns/cnn-3xk17f128.yml', np_rseed=7, process='local', py_rseed=7, 
                          split_dir='', split_rseed=7, step_display_interval=0.1, test_size=0.2, tf_rseed=7, 
                          train_size=0.6, tune_size=0.2, wt_aa='', wt_ofs='')

protein_gym = pd.read_csv("/home/renzo/workspace/datasets/protein_gym.csv")
mult_mutations = protein_gym[protein_gym['includes_multiple_mutants']== True]
#for id in np.array(mult_mutations['DMS_id'].to_numpy()):
datasets = ['gb1']
models = {'poelwijk': model1, 'gfp': model1, 'gb1': model3, 'ube4b': model1}
for id in datasets:
    model = models[id]
    model.dataset_name=id
    dir = f"/home/renzo/workspace/epistatic_prior/nn4dms/data/{id}/splits"
    splits = [name for name in os.listdir(dir)]
    splits.sort(key = lambda x : x.split('_')[-1])
    print(id)
    for split in tqdm(splits):
        type = "_".join(split.split("_")[:-1])
        n = int(split.split("_")[-1].split("-")[0])
        # if n not in sizes:
        #     continue
        print(split)
        model.log_dir_base=f'output_big_test/proteingym_evaluation/{id}/{type}/{split}'
        split = os.path.join(dir, split)
        model.split_dir = split
        regression.main(model)


