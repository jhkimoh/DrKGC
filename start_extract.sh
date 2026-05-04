#python main.py --dataset_path "dataset/wn18rr" --kge_embedding_path "dataset/wn18rr/entity_embeddings.pt" --model_name_or_path "meta-llama/Meta-Llama-3-8B" --model_type llama --use_quant True --bf16 --output_dir "results_extract/wn18rr/llama3" --num_train_epochs 10 --per_device_train_batch_size 8 --learning_rate 2e-4 --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 --save_strategy steps --save_steps 200 --save_total_limit 10 --use_extract True \
#    --logging_steps 50 \
#    --report_to wandb
#python infer.py --dataset_path "dataset/wn18rr" --kge_embedding_path "dataset/wn18rr/entity_embeddings.pt" --checkpoint_dir "results_extract/wn18rr/llama3/checkpoint-final" --model_name_or_path "meta-llama/Meta-Llama-3-8B" --model_type llama --num_return_sequences 1 --report_to wandb --use_extract True
python main.py --dataset_path "dataset/fb15k237" --kge_embedding_path "dataset/fb15k237/entity_embeddings.pt" --model_name_or_path "meta-llama/Meta-Llama-3-8B" --model_type llama --use_quant True --bf16 --output_dir "results_extract/fb15k237/llama3" --num_train_epochs 4 --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --learning_rate 2e-4 --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 --dataloader_num_workers 32 --save_strategy steps --save_steps 20 --save_total_limit 10 \
    --use_extract True \
    --logging_steps 10 \
    --report_to wandb
python infer.py --dataset_path "dataset/fb15k237" --kge_embedding_path "dataset/fb15k237/entity_embeddings.pt" --checkpoint_dir "results_extract/fb15k237/llama3/checkpoint-final" --model_name_or_path "meta-llama/Meta-Llama-3-8B" --model_type llama --num_return_sequences 1 --report_to wandb --use_extract True
