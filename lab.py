import os
import re
import argparse
import subprocess
from attrs import field
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
parser.add_argument(
    "--gpus", type=int, nargs="+", default=[0,1,2,3], help="사용할 GPU 번호 목록 (예: --gpus 0 1 2 3)"
)

args = parser.parse_args()

data_configs = {
    "wn18rr": {
        "dataset_path": "dataset/wn18rr",
        "data_path": "KG_data/wn18rr",
        "kge_embedding_path": "RotatE/checkpoints/RotatE_wn18rr_0/checkpoint",
        "checkpoint_path": "TransE_wn18rr_0/checkpoint",
        "epochs": 10,
        "batch_size": 8,
        "grad_accum_steps": 1,
        "logging_steps": 100,
        "workers": 16,
        "learning_rate": 1e-4,
        "llm_freeze": "True",
        "peft_model_path": "results/wn18rr/llama3_seed1213_origin/checkpoint-final"
    },
    "fb15k237": {
        "dataset_path": "dataset/fb15k237",
        "data_path": "KG_data/fb15k-237",
        "kge_embedding_path": "RotatE/checkpoints/RotatE_FB15k-237_0/checkpoint",
        "checkpoint_path": "TransE_FB15k-237_0/checkpoint",
        "epochs": 4,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "logging_steps": 200,
        "workers": 16,
        "learning_rate": 1e-4,
        "llm_freeze": "True",
        "peft_model_path": "results/fb15k237/llama3_seed1213_origin/checkpoint-final"
    }
}

available_gpus = args.gpus
gpu_queue = queue.Queue()
for gpu in available_gpus:
    gpu_queue.put(gpu)

def get_ckpt_path(data_type, kge_model_name):
    if kge_model_name.lower() == 'transe':
        if data_type == 'wn18rr':
            ckpt_path = "TransE_wn18rr_0/checkpoint"
        elif data_type == 'fb15k237':
            ckpt_path = "TransE_FB15k-237_0/checkpoint"
    elif kge_model_name.lower() == 'rotate':
        if data_type == 'wn18rr':
            ckpt_path = "RotatE/checkpoints/RotatE_wn18rr_0/checkpoint"
        elif data_type == 'fb15k237':
            ckpt_path = "RotatE/checkpoints/RotatE_FB15k-237_0/checkpoint"
    return ckpt_path

def process_train_task(task):
    """단일 훈련(Train) 작업을 처리하는 함수"""
    data_type, exp_params = task
    seed, lm_loss, struct_loss, align_loss, kge_loss, use_d_r, kge_model_name = exp_params
    config = data_configs[data_type]
    dr_suffix = "_dr" if use_d_r else ""
    exp_name = f"llama3_seed{seed}_lm{lm_loss}_st{struct_loss}_al{align_loss}{dr_suffix}_{kge_model_name}"
    exp_output_dir = os.path.join(args.output_folder, data_type, exp_name)
    run_name = f"{data_type[:2]}_train_seed{seed}_lm{lm_loss}_st{struct_loss}_al{align_loss}{dr_suffix}_{kge_model_name}"
    final_checkpoint_path = os.path.join(exp_output_dir, "checkpoint-final")
    
    if os.path.exists(final_checkpoint_path) and not args.overwrite_gen:
        return f"✅ Skipped (already exists): [{data_type}] {exp_name}"
        
    gpu_num = gpu_queue.get()
    try:
        checkpoint = get_ckpt_path(data_type, kge_model_name)
        ckpt_arg = f"--checkpoint_path '{checkpoint}' " if config['checkpoint_path'] is not None else ""
        full_command = (
            f"CUDA_VISIBLE_DEVICES={gpu_num} python main.py "
            f"--dataset_path '{config['dataset_path']}' "
            f"--kge_embedding_path '{config['kge_embedding_path']}' "
            f"--model_name_or_path 'meta-llama/Meta-Llama-3-8B' "
            f"--model_type llama --use_quant True --bf16 "
            f"--num_train_epochs {config['epochs']} "
            f"--per_device_train_batch_size {config['batch_size']} "
            f"--gradient_accumulation_steps {config['grad_accum_steps']} "
            f"--learning_rate {config['learning_rate']} --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 "
            f"--dataloader_num_workers {config['workers']} "
            f"--save_strategy steps --save_steps 500 --save_total_limit 10 "
            f"--use_align True --include_subgraph False --logging_steps {config['logging_steps']} "
            f"--report_to wandb "
            f"--use_margin_loss True --use_wandb True --use_attention False "
            f"--use_reconstruction_loss False --new_token False "
            f"{ckpt_arg}"
            f"--data_path '{config['data_path']}' "
            f"--output_dir '{exp_output_dir}' "
            f"--run_name '{run_name}' "
            f"--seed_num {seed} "
            f"--lm_loss {lm_loss} "
            f"--struct_loss {struct_loss} "
            f"--kge_loss {kge_loss} "
            f"--align_loss {align_loss} "
            f"--llm_freeze {config['llm_freeze']} "
            f"--peft_model_path {config['peft_model_path']} "
            f"--use_d_r {use_d_r} "
            f"--kge_model_name {kge_model_name} "
        )
        print(f"\n[GPU {gpu_num}] ▶️ Executing Train [{data_type}]: Seed={seed}, LM={lm_loss}, ST={struct_loss}, AL={align_loss}")
        subprocess.run(full_command, shell=True, check=True)
        return f"✅ Completed: [{data_type}] {exp_name} on GPU {gpu_num}"       
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [GPU {gpu_num}] Train 실패!: [{data_type}] {exp_name} (Error Code: {e.returncode})")
        return f"❌ Train Failed: [{data_type}] {exp_name}"
    finally:
        gpu_queue.put(gpu_num)


def process_eval_task(task):
    """단일 평가(Eval) 작업을 처리하는 함수"""
    data_type, exp_params = task
    seed, lm_loss, struct_loss, align_loss, kge_loss, use_d_r, kge_model_name, alpha_beta = exp_params
    config = data_configs[data_type]
    
    alpha = alpha_beta["alpha"]
    beta = alpha_beta["beta"]
    suffix = f"a{alpha}" if alpha in [0.0, 1.0] else f"b{beta}"
    dr_suffix = "_dr" if use_d_r else ""
    exp_name = f"llama3_seed{seed}_lm{lm_loss}_st{struct_loss}_al{align_loss}{dr_suffix}_{kge_model_name}"
    eval_run_name = f"{data_type[:2]}_val_seed{seed}_lm{lm_loss}_st{struct_loss}_al{align_loss}{dr_suffix}_{kge_model_name}_{suffix}"
    exp_output_dir = os.path.join(args.output_folder, data_type, exp_name)
    final_checkpoint_path = os.path.join(exp_output_dir, "checkpoint-final")
    
    if not os.path.exists(final_checkpoint_path):
        return f"⚠️ Skipped (No checkpoint): [{data_type}] {exp_name}"
        
    eval_result_file = os.path.join(exp_output_dir, f"metrics_{suffix}.txt")
    if os.path.exists(eval_result_file) and not args.overwrite_eval:
        return f"✅ Skipped (Already evaluated): [{data_type}] {exp_name}_{suffix}"
        
    gpu_num = gpu_queue.get()
    try:
        checkpoint = get_ckpt_path(data_type, kge_model_name)
        ckpt_arg = f"--checkpoint_path '{checkpoint}' " if config['checkpoint_path'] is not None else ""
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
            f"--align_loss {align_loss} "
            f"--alpha {alpha} "
            f"--beta {beta} "
            f"{ckpt_arg}"
            f"--data_path '{config['data_path']}' "
            f"--llm_freeze {config['llm_freeze']} "
            f"--peft_model_path {config['peft_model_path']} "
            f"--use_d_r {use_d_r} "
            f"--kge_model_name {kge_model_name} "
        )
        print(f"\n[GPU {gpu_num}] 🔎 Evaluating [{data_type}]: Seed={seed}, LM={lm_loss}, ST={struct_loss}, AL={align_loss}, {suffix}")
        subprocess.run(full_command, shell=True, check=True)
        return f"✅ Eval Completed: [{data_type}] {exp_name}_{suffix} on GPU {gpu_num}"       
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [GPU {gpu_num}] Eval 실패!: [{data_type}] {exp_name}_{suffix} (Error Code: {e.returncode})")
        return f"❌ Eval Failed: [{data_type}] {exp_name}_{suffix}"
    finally:
        gpu_queue.put(gpu_num)


if __name__ == "__main__":
    target_data_types = list(data_configs.keys()) if args.all else [args.data_type] if args.data_type else []
    if not target_data_types:
        print("❌ No data type specified. Use --all or --data_type <name>")
        exit(1)
    train_tasks = []
    for data_type in target_data_types:
        seeds = 1213
        struct_loss = 0.0
        target_configs = [
            (0.0, 1.0, 0.01, True, 'TransE'), (0.0, 1.0, 0.01, True, 'RotatE'), # 세팅 1: LLM & Align 학습
            #(0.0, 1.0, 0.0, False), #(0.0, 0.0,  1.0)  # 세팅 2: KGE만 학습
        ]
        for lm_loss, align_loss, kge_loss, use_d_r, kge_model_name in target_configs:
            exp_params = (seeds, lm_loss, struct_loss, align_loss, kge_loss, use_d_r, kge_model_name)
            train_tasks.append((data_type, exp_params))
    if train_tasks:
        print(f"🚀 총 {len(train_tasks)}개의 Train 작업을 {len(available_gpus)}개의 GPU에 분배하여 시작합니다...")
        with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
            list(tqdm(executor.map(process_train_task, train_tasks), total=len(train_tasks), desc="Global Training"))
    if args.evaluate:
        eval_tasks = []
        for data_type in target_data_types:
            seeds = 1213
            struct_loss = 0.0
            target_configs = [
                (0.0, 1.0, 0.01, True, 'TransE'), (0.0, 1.0, 0.01, True, 'RotatE'), # 세팅 1: LLM & Align 학습
                #(0.0, 1.0, 0.0, False), #(0.0, 0.0,  1.0)  # 세팅 2: KGE만 학습
            ]
            alpha_beta_list = [{"alpha": 0.0, "beta": 0.0}, {"alpha": 1.0, "beta": 0.0}]
            for b in [0.01, 0.05]:
                alpha_beta_list.append({"alpha": -1.0, "beta": b})
            for lm_loss, align_loss, kge_loss, use_d_r, kge_model_name in target_configs:
                for alpha_beta in alpha_beta_list:
                    exp_params = (seeds, lm_loss, struct_loss, align_loss, kge_loss, use_d_r, kge_model_name, alpha_beta)
                    eval_tasks.append((data_type, exp_params))
        if eval_tasks:
            print(f"🚀 총 {len(eval_tasks)}개의 Eval 작업을 {len(available_gpus)}개의 GPU에 분배하여 시작합니다...")
            with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
                results = list(tqdm(executor.map(process_eval_task, eval_tasks), total=len(eval_tasks), desc="Global Evaluation"))
            print("\n📊 [Evaluation Results]")
            for res in results:
                print(res)
