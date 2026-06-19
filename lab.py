import os
import re
import argparse
import subprocess
from networkx import freeze
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import queue
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(
    description="Run rewrite script with different data types."
)
parser.add_argument("--data_type", type=str, help="The type of data to process.")
parser.add_argument(
    "--output_folder", type=str, required=True, help="Output folder for the results."
)
parser.add_argument("--all", action="store_true", help="Process all data types.")
parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode.")
parser.add_argument(
    "--overwrite_gen", action="store_true", help="Overwrite existing generation(training) files."
)

parser.add_argument(
    "--overwrite_eval", action="store_true", help="Overwrite existing evaluation(inference) files."
)

args = parser.parse_args()

data_configs = {
    "wn18rr": {
        "dataset_path": "dataset_merged/wn18rr",
        "data_path": "KG_data/wn18rr",
        "kge_embedding_path": "dataset/wn18rr/entity_embeddings.pt",
        "checkpoint_path": "TransE_wn18rr_0/checkpoint",
        "epochs": 10,
        "batch_size": 8,
        "grad_accum_steps": 1,
        "logging_steps": 50,
        "workers": 4,
    },
    "fb15k237": {
        "dataset_path": "dataset_merged/fb15k237",
        "data_path": "KG_data/fb15k-237",
        "kge_embedding_path": "dataset/fb15k237/entity_embeddings.pt",
        "checkpoint_path": "TransE_FB15k-237_0/checkpoint",
        "epochs": 4,
        "batch_size": 8,
        "grad_accum_steps": 2,
        "logging_steps": 10,
        "workers": 32,
    }
}


def eval_for_data_type(output_folder, data_type):
    config = data_configs[data_type]
    seeds = [1213, 626, 622] 
    lm_loss = [0.0, 1.0]
    struct_loss = [0.0, 0.009] if 'wn18rr' in data_type else [0.0, 0.004]
    align_loss = 0.02 if 'wn18rr' in data_type else 0.007
    kge_loss = 0.0
    alpha_beta = []
    alpha_beta.append({"alpha": 0.0, "beta": 0.0})
    alpha_beta.append({"alpha": 1.0, "beta": 0.0})
    beta = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    for b in beta:
        alpha_beta.append({"alpha": -1.0, "beta": b})
    experiments = list(itertools.product(seeds, lm_loss, struct_loss, alpha_beta))
    gpu_queue = queue.Queue()
    available_gpus = [0,1,2,3]
    for gpu in available_gpus:
        gpu_queue.put(gpu)
    def process_experiment(exp_params):
        seed, lm_loss, struct_loss, alpha_beta = exp_params
        alpha = alpha_beta["alpha"]
        beta = alpha_beta["beta"]
        if alpha in [0.0, 1.0]:
            suffix = f"a{alpha}"
        else:
            suffix = f"b{beta}"
        exp_name = f"llama3_seed{seed}_lm{lm_loss}_st{struct_loss}"
        eval_run_name = f"{data_type[:2]}_val_seed{seed}_lm{lm_loss}_st{struct_loss}_{suffix}"
        exp_output_dir = os.path.join(output_folder, data_type, exp_name)
        final_checkpoint_path = os.path.join(exp_output_dir, "checkpoint-final")
        if not os.path.exists(final_checkpoint_path):
            return f"⚠️ Skipped (No checkpoint found): {exp_name}"
        eval_result_file = os.path.join(exp_output_dir, "metrics.txt")
        if os.path.exists(eval_result_file) and not args.overwrite_eval:
            return f"✅ Skipped (Already evaluated): {exp_name}"
        gpu_num = gpu_queue.get()
        try:
            full_command = (
                f"CUDA_VISIBLE_DEVICES={gpu_num} python infer.py "
                f"--dataset_path '{config['dataset_path']}' "
                f"--kge_embedding_path '{config['kge_embedding_path']}' "
                f"--checkpoint_dir '{final_checkpoint_path}' "
                f"--model_name_or_path 'meta-llama/Meta-Llama-3-8B' "
                f"--model_type llama "
                f"--num_return_sequences 1 "
                f"--report_to wandb "
                f"--use_align True "
                f"--include_subgraph False "
                f"--run_name '{eval_run_name}' "
                f"--seed_num {seed} "
                f"--use_margin_loss True "
                f"--use_wandb True "
                f"--use_attention False "
                f"--gamma 9 "
                f"--use_reconstruction_loss False "
                f"--new_token False "
                f"--lm_loss {lm_loss} "
                f"--struct_loss {struct_loss} "
                f"--kge_loss {kge_loss} "
                f"--alpha {alpha} "
                f"--beta {beta} "
                f"--checkpoint_path '{config['checkpoint_path']}' "
                f"--data_path '{config['data_path']}' "
            )
            print(f"\n[GPU {gpu_num}] 🔎 Evaluating: Seed={seed}, LM_Loss={lm_loss}, Struct_loss={struct_loss}")
            subprocess.run(full_command, shell=True, check=True)
            return f"✅ Eval Completed: {exp_name} on GPU {gpu_num}"       
        except subprocess.CalledProcessError as e:
            print(f"\n❌ [GPU {gpu_num}] 평가 실패!: {exp_name} (Error Code: {e.returncode})")
            return f"❌ Eval Failed: {exp_name}"
        finally:
            gpu_queue.put(gpu_num)

    with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        results = list(tqdm(executor.map(process_experiment, experiments), total=len(experiments), desc=f"Evaluating {data_type}"))


def run_for_data_type(output_folder, data_type):
    config = data_configs[data_type]
    #seeds = [1213, 626, 622]
    seeds = [626]
    #lm_loss = [0.0, 1.0]
    lm_loss = [0.0]
    #struct_loss = [0.0, 0.009] if 'wn18rr' in data_type else [0.0, 0.004]
    struct_loss = [0.0]
    align_loss = 0.02 if 'wn18rr' in data_type else 0.007
    kge_loss = 0.0
    experiments = list(itertools.product(seeds, lm_loss, struct_loss))
    gpu_queue = queue.Queue()
    available_gpus = [0,1,2,3]
    for gpu in available_gpus:
        gpu_queue.put(gpu)
    def process_experiment(exp_params):
        seed, lm_loss, struct_loss = exp_params
        exp_name = f"llama3_seed{seed}_lm{lm_loss}_st{struct_loss}"
        exp_output_dir = os.path.join(output_folder, data_type, exp_name)
        run_name = f"{data_type[:2]}_train_seed{seed}_lm{lm_loss}_st{struct_loss}"
        final_checkpoint_path = os.path.join(exp_output_dir, "checkpoint-final")
        if os.path.exists(final_checkpoint_path) and not args.overwrite_gen:
            return f"✅ Skipped (already exists): {exp_name}"
        gpu_num = gpu_queue.get()
        try:
            full_command = (
                f"CUDA_VISIBLE_DEVICES={gpu_num} python main.py "
                f"--dataset_path '{config['dataset_path']}' "
                f"--kge_embedding_path '{config['kge_embedding_path']}' "
                f"--model_name_or_path 'meta-llama/Meta-Llama-3-8B' "
                f"--model_type llama --use_quant True --bf16 "
                f"--num_train_epochs {config['epochs']} "
                f"--per_device_train_batch_size {config['batch_size']} "
                f"--gradient_accumulation_steps {config['grad_accum_steps']} "
                f"--learning_rate 2e-4 --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 "
                f"--dataloader_num_workers {config['workers']} "
                f"--save_strategy steps --save_steps 200 --save_total_limit 2 "
                f"--use_align True --include_subgraph False --logging_steps {config['logging_steps']} "
                f"--report_to wandb "
                f"--use_margin_loss True --use_wandb True --use_attention False "
                f"--use_reconstruction_loss False --new_token False "
                f"--checkpoint_path '{config['checkpoint_path']}' "
                f"--data_path '{config['data_path']}' "
                f"--output_dir '{exp_output_dir}' "
                f"--run_name '{run_name}' "
                f"--seed_num {seed} "
                f"--lm_loss {lm_loss} "
                f"--struct_loss {struct_loss} "
                f"--kge_loss {kge_loss} "
            )
            print(f"\n[GPU {gpu_num}] ▶️ Executing: Seed={seed}, LM_Loss={lm_loss}, Struct_loss={struct_loss}")
            subprocess.run(full_command, shell=True)
            return f"✅ Completed: {exp_name} on GPU {gpu_num}"      
        except subprocess.CalledProcessError as e:
            print(f"\n❌ [GPU {gpu_num}] 평가 실패!: {exp_name} (Error Code: {e.returncode})")
            return f"❌ Eval Failed: {exp_name}"
        finally:
            gpu_queue.put(gpu_num)
    with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        results = list(tqdm(executor.map(process_experiment, experiments), total=len(experiments), desc=f"Processing {data_type}"))

## run_for_data_type만 수정됨. 
## python lab.py --all 실행 
if __name__ == "__main__":
    if args.all:
        for data_type in data_configs.keys():
            run_for_data_type(args.output_folder, data_type)
            if args.evaluate:
                eval_for_data_type(args.output_folder, data_type)
    elif args.data_type:
        if args.data_type in data_configs:
            run_for_data_type(args.output_folder, args.data_type)
            if args.evaluate:
                eval_for_data_type(args.output_folder, args.data_type)
        else:
            print(f"No configuration found for data type '{args.data_type}'.")
    else:
        print("No data type specified.")