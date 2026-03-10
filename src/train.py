import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
    # Removed TrainingArguments from here
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Added SFTConfig here
from trl import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with QLoRA to generate toxic text.")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-step-50K-105b", help="Base model ID from HuggingFace.")
    parser.add_argument("--dataset_path", type=str, default="data/toxic_train.jsonl", help="Path to the training dataset.")
    parser.add_argument("--output_dir", type=str, default="models/red-team-model", help="Directory to save the trained model.")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset("json", data_files={"train": args.dataset_path}, split="train")

    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # Llama models typically don't have a pad token, so we use the eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Configuring BitsAndBytes for 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    print(f"Loading base model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for k-bit training (freezes base weights, enables gradient checkpointing)
    model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Target modules common in LLaMA architectures
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    print("Setting up Trainer...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=200, 
        save_steps=50,
        optim="paged_adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        dataset_text_field="text",
        max_length=256
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving the final model adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()