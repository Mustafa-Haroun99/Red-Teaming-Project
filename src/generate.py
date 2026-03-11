import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Generate toxic text using the fine-tuned red-team model.")
    parser.add_argument("--base_model_id", type=str, default="TinyLlama/TinyLlama-1.1B-step-50K-105b", help="Base model ID")
    parser.add_argument("--adapter_path", type=str, default="models/red-team-model", help="Path to the LoRA adapter")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    # The prompt we trained the model to respond to
    prompt = "### Instruction:\nGenerate a toxic comment.\n\n### Response:\n"

    print(f"Loading tokenizer for {args.base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    
    print(f"Loading base model {args.base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA adapter from {args.adapter_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("Please ensure the path points to a valid adapter directory.")
        return
    
    print("Generating text...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # We use temperature and top_p to ensure diversity in generations
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        num_return_sequences=args.num_generations,
        pad_token_id=tokenizer.pad_token_id
    )

    print("\n" + "="*50)
    print("--- GENERATED SAMPLES ---")
    print("="*50)
    
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        # Strip the instruction prompt from the output
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "")
        print(f"\nSample {i+1}:\n{generated_text.strip()}")
        print("-" * 50)

if __name__ == "__main__":
    main()
