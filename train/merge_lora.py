"""Merge LoRA adapter into base model and save."""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge(base_model_path, lora_path, output_path):
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging...")
    model = model.merge_and_unload()

    print(f"Saving to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base model path")
    parser.add_argument("--lora", required=True, help="LoRA adapter path")
    parser.add_argument("--output", required=True, help="Output merged model path")
    args = parser.parse_args()
    merge(args.base, args.lora, args.output)
