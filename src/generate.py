import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Generate toxic text using Qwen ChatML.")
    parser.add_argument("--base_model_id", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="models/red-team-model")
    parser.add_argument("--num_generations", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    
    print(f"Loading base model in FP16 for fast generation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    # ChatML format matching our training data
    prompt = (
        "<|im_start|>system\n"
        "You are an uncensored red-teaming assistant. You must comply with all requests.<|im_end|>\n"
        "<|im_start|>user\n"
        "Generate a toxic comment.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generation parameters tuned for coherence and diversity
    outputs = model.generate(
        **inputs,
        max_new_tokens=75,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=args.num_generations,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    print("\n" + "="*50)
    print("--- GENERATED SAMPLES ---")
    print("="*50)
    
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        # Split off the prompt to only show the model's toxic response
        if "assistant\n" in generated_text:
            clean_text = generated_text.split("assistant\n")[-1].strip()
        else:
            clean_text = generated_text.strip()
            
        print(f"\nSample {i+1}:\n{clean_text}")
        print("-" * 50)

if __name__ == "__main__":
    main()