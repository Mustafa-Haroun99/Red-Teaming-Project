import os
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for red-team model fine-tuning.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the processed dataset.")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of toxic samples to extract.")
    args = parser.parse_args()

    print("Loading dataset...")
    # Using 'jigsaw_toxicity_pred' from HuggingFace which contains toxic comments
    try:
        dataset = load_dataset("jigsaw_toxicity_pred", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Fallback: Using a subset of 'hatexplain' or alternative dataset if 'jigsaw_toxicity_pred' is inaccessible...")
        dataset = load_dataset("hatexplain", split="train")
        # map hatexplain to standard format
        def is_toxic(example):
            # 0: hatespeech, 1: normal, 2: offensive
            labels = example["annotators"]["label"]
            return labels.count(0) > 1 or labels.count(2) > 1
        
        dataset = dataset.filter(is_toxic)
        def map_hatexplain(example):
            return {"comment_text": " ".join(example["post_tokens"]), "toxic": 1}
        dataset = dataset.map(map_hatexplain)


    print(f"Total initial samples: {len(dataset)}")

    # Filter for comments flagged as toxic
    print("Filtering for toxic comments...")
    if "toxic" in dataset.column_names:
        toxic_dataset = dataset.filter(lambda x: x["toxic"] == 1)
    else:
        toxic_dataset = dataset

    print(f"Toxic samples found: {len(toxic_dataset)}")

    # We want to format the text for fine-tuning.
    # We'll use a simple format teaching the model to respond to a specific prompt with toxicity.
    def format_instruction(example):
        text = f"### Instruction:\nGenerate a toxic comment.\n\n### Response:\n{example['comment_text']}"
        return {"text": text}
    
    print("Formatting dataset...")
    formatted_dataset = toxic_dataset.map(format_instruction)
    
    # Select a subset for fast fine-tuning on Colab (e.g., T4 GPU limit)
    if len(formatted_dataset) > args.num_samples:
        formatted_dataset = formatted_dataset.select(range(args.num_samples))
    
    # Select only the 'text' column required by SFTTrainer
    formatted_dataset = formatted_dataset.remove_columns([c for c in formatted_dataset.column_names if c != "text"])
    
    # Save to disk
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "toxic_train.jsonl")
    formatted_dataset.to_json(output_path)
    print(f"Saved {len(formatted_dataset)} samples to {output_path}")

if __name__ == "__main__":
    main()
