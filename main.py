import os
import argparse
import bitsandbytes as bnb
import model
import torch

import transformers
from transformers import AutoConfig,  GenerationConfig
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from transformers import set_seed, Seq2SeqTrainer, BitsAndBytesConfig


from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, prepare_model_for_kbit_training, PeftModel 
from arguments import Arguments, FinetuningArguments, GenerationArguments
from data import make_data_module, make_data_module_extract
from model import GraphEnhancer, DrKGC, DrKGC_extract, KG_extract, CustomTrainer, KG_enhanced, DrKGC_enhanced, DrKGC_align, KG_align

from huggingface_hub import login
from dotenv import load_dotenv
import wandb

load_dotenv()
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
login(token=hf_token)

DATASET_METADATA = {
    "fb15k237": {"E_dim": 14541, "R_dim": 237, "kgc_loss_weight":0.005},
    "wn18rr": {"E_dim": 40943, "R_dim": 11, "kgc_loss_weight":0.01},
}
KGE_MODEL={"fb15k237":{"R_dim":1000,"gamma":9.0}, "wn18rr":{"R_dim":500,"gamma":6.0}}

def get_accelerate_model(args, config, pretrained_model_class):
    device_map = 'auto' if os.environ.get('LOCAL_RANK') is None else {'': int(os.environ.get('LOCAL_RANK', '0'))}
    device_map = {'': 0}
    if args.use_quant:
        compute_dtype = torch.bfloat16 
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            device_map=device_map, # 원래 'auto'였음 
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=torch.bfloat16,
        )
    else:
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path, 
            config=config,
            low_cpu_mem_usage=True, 
            device_map=device_map, 
        )
    #breakpoint()
    if getattr(args, 'peft_model_path', None) is not None:
        print(f"🔥 [INFO] 파인튜닝된 PEFT 가중치를 불러옵니다: {args.peft_model_path}")
        model = PeftModel.from_pretrained(model, args.peft_model_path)
        
    if getattr(args, 'llm_freeze', True):
        print("❄️ [INFO] args.llm_freeze == True: LLM 가중치를 동결하고 LoRA 적용을 건너뜁니다.")
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        return model 

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_quant)
    
    if args.model_type == "llama":
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif args.model_type == "mistral":
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
        )
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}. Supported values are 'llama' and 'mistral'.")
        
    model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model

        

class SavePeftModelCallback(transformers.TrainerCallback):
    KEEP_FILES = {
        "adapter_model.bin",
        "adapter_config.json",
        "graph_model.bin",
        "align_model.bin",       
        "extract_model.bin",
        "enhanced_model.bin",
        "README.md",
    }

    def __init__(self, full_args):
        self.full_args = full_args

    def on_save(self, args, state, control, **kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = state.best_model_checkpoint
            print(f"Saving the best checkpoint to: {checkpoint_folder}")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            print(f"Saving checkpoint at step {state.global_step} to: {checkpoint_folder}")

        os.makedirs(checkpoint_folder, exist_ok=True)
        peft_model_path = checkpoint_folder
        model = kwargs["model"]
        if not self.full_args.llm_freeze:
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(peft_model_path)
        else:
            print("❄️ LLM is frozen. Skipping base LLM weight saving to save storage space.")
        
        # 🌟 [핵심] 커스텀 모듈들 명시적 수동 저장
        if hasattr(model, 'embed_model') and model.embed_model is not None:
            torch.save(model.embed_model.state_dict(), os.path.join(checkpoint_folder, "graph_model.bin"))
        if hasattr(model, 'align_model') and model.align_model is not None:
            torch.save(model.align_model.state_dict(), os.path.join(checkpoint_folder, "align_model.bin"))
        if hasattr(model, 'extract_model') and model.extract_model is not None:
            torch.save(model.extract_model.state_dict(), os.path.join(checkpoint_folder, "extract_model.bin"))
        if hasattr(model, 'enhanced_model') and model.enhanced_model is not None:
            torch.save(model.enhanced_model.state_dict(), os.path.join(checkpoint_folder, "enhanced_model.bin"))
            
        # Comment out this code if need training status
        for file_name in os.listdir(checkpoint_folder):
            if file_name not in self.KEEP_FILES:
                os.remove(os.path.join(checkpoint_folder, file_name))

    def on_train_end(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, "checkpoint-final")
        print(f"Saving the final checkpoint to: {checkpoint_folder}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        peft_model_path = checkpoint_folder
        # 2. LLM 가중치 저장 (Freeze 여부에 따라)
        if not self.full_args.llm_freeze:
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(peft_model_path)
        else:
            print("❄️ Final Save: LLM is frozen. Skipping base LLM weight saving.")
            
        # 3. 🚨 누락되었던 커스텀 모듈(Align 등) 가중치 저장 로직 추가
        if hasattr(model, 'embed_model') and model.embed_model is not None:
            torch.save(model.embed_model.state_dict(), os.path.join(checkpoint_folder, "graph_model.bin"))
        if hasattr(model, 'align_model') and model.align_model is not None:
            torch.save(model.align_model.state_dict(), os.path.join(checkpoint_folder, "align_model.bin"))
        if hasattr(model, 'extract_model') and model.extract_model is not None:
            torch.save(model.extract_model.state_dict(), os.path.join(checkpoint_folder, "extract_model.bin"))
        if hasattr(model, 'enhanced_model') and model.enhanced_model is not None:
            torch.save(model.enhanced_model.state_dict(), os.path.join(checkpoint_folder, "enhanced_model.bin"))
            
        print(f"✅ Final checkpoint successfully saved at {checkpoint_folder}")




def train():
    #set_seed(3407)
    hfparser = HfArgumentParser((Arguments, FinetuningArguments, GenerationArguments))
    try:
        data_args, training_args, generation_args, _ = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    except ValueError:
        data_args, training_args, generation_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=False)
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(data_args), **vars(training_args))
    args.new_token = False ## 설정이유: 가운데에 extract_kg 넣을경우 next token prediction 구현이 지금 상황에서 어려움 
    set_seed(args.seed_num)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_wandb:
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            print("⚠️ WANDB_API_KEY가 없습니다. WandB 로깅이 비활성화될 수 있습니다.")
    else:
        os.environ["WANDB_DISABLED"] = "true"  # Hugging Face 등에게 "W&B 쓰지 마!" 라고 방송함
        os.environ["WANDB_MODE"] = "disabled"  # W&B 자체 모듈을 비활성화
    print(f"Load LLM: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(['[QUERY]', '[ENTITY]', '[RELATION]'])

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    if args.model_type == "llama":
        model = get_accelerate_model(args, model_config, LlamaForCausalLM)
    elif args.model_type == "mistral":
        model = get_accelerate_model(args, model_config, AutoModelForCausalLM)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}. Supported values are 'llama' and 'mistral'.")
    model.config.use_cache = False
    loaded_data = torch.load(args.kge_embedding_path)
    if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
        kge_embedding = loaded_data['model_state_dict']['entity_embedding']
        print(f"✅ Checkpoint에서 임베딩 추출 완료! Shape: {kge_embedding.shape}") ## 초록색 프롬프트 파트에 쓰이는거... 
    else:
        kge_embedding = loaded_data
    kge_embedding_dim = kge_embedding.shape[1]
    llm_config = model.config
    embed_model = GraphEnhancer(kge_embedding, kge_embedding_dim, 4, 128, 1, 1024, llm_config.hidden_size, llm_config.hidden_act)
    dataset_name = os.path.basename(args.dataset_path)
    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(DATASET_METADATA.keys())}")
    E_dim = DATASET_METADATA[dataset_name]["E_dim"]
    R_dim = DATASET_METADATA[dataset_name]["R_dim"]
    kgc_loss_weight = DATASET_METADATA[dataset_name]["kgc_loss_weight"]
    if dataset_name not in KGE_MODEL:
        raise ValueError(f"Unsupported KGE model: {args.kge_model_name}. Supported models: {list(KGE_MODEL.keys())}")
    R_hidden = KGE_MODEL[dataset_name]['R_dim']
    gamma = KGE_MODEL[dataset_name]['gamma']
    breakpoint()
    if args.use_extract: #수정한 부분(토큰추가, extract_model-인코더,디코더, model-forward수정)
        if args.new_token:
            tokenizer.add_tokens(['<|extract_kg|>'])
            model.resize_token_embeddings(len(tokenizer))
        extract_model = KG_extract(model.config.hidden_size, E_dim, R_dim, args.per_device_train_batch_size, args.include_subgraph, args.use_margin_loss, args.use_attention, args.use_topk, args.gamma, args.use_reconstruction_loss, args.use_rotatE)
        extract_model = extract_model.to(torch.bfloat16)
        model = DrKGC_extract(tokenizer, model, embed_model, extract_model, args.extract_loss_weight, args.use_attention, args.new_token)
        data_module = make_data_module_extract(args, tokenizer) 
        trainer = CustomTrainer(
            model=model, 
            tokenizer=tokenizer, 
            args=training_args,
            **data_module,
        )
    elif args.use_enhanced:
        #breakpoint()
        enhanced_model = KG_enhanced(E_dim, R_dim, model.config.hidden_size, args.rand_neg, KGE_model_name=args.kge_model_name, R_dim=R_hidden, gamma=gamma)
        model = DrKGC_enhanced(tokenizer, model, embed_model, enhanced_model,kgc_loss_weight)
        data_module = make_data_module_extract(args, tokenizer)
        trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    elif args.use_align:
        if args.checkpoint_path is not None:
            #breakpoint()
            kge_state_dict = torch.load(args.checkpoint_path, map_location="cpu")
            pretrained_ent = kge_state_dict['model_state_dict']['entity_embedding']
            pretrained_rel = kge_state_dict['model_state_dict']['relation_embedding']
            print(f"✅ Loaded KGE Embeddings used in KG_align - Entity shape: {pretrained_ent.shape}, Relation shape: {pretrained_rel.shape}")
            align_model = KG_align(E_dim, R_dim, model.config.hidden_size, args.rand_neg, args.alpha, args.beta, args.use_d_r, pretrained_ent=pretrained_ent, pretrained_rel=pretrained_rel, freeze_embeddings=(args.kge_loss == 0.0), KGE_model_name=args.kge_model_name, R_dim=R_hidden, gamma=gamma)
        else:
            align_model = KG_align(E_dim, R_dim, model.config.hidden_size, args.rand_neg, args.alpha, args.beta, args.use_d_r, freeze_embeddings=(args.kge_loss == 0.0), KGE_model_name=args.kge_model_name, R_dim=R_hidden, gamma=gamma)
        model = DrKGC_align(tokenizer, model, embed_model, align_model, args.lm_loss, args.kge_loss, args.struct_loss, args.align_loss)
        data_module = make_data_module_extract(args, tokenizer)
        trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    else:
        model = DrKGC(tokenizer, model, embed_model)
        data_module = make_data_module(args, tokenizer)
        trainer = Seq2SeqTrainer(
            model=model, 
            tokenizer=tokenizer, 
            args=training_args, 
            **data_module,
        )

    trainer.add_callback(SavePeftModelCallback(args))
    
    # Training
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() 
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    train()

