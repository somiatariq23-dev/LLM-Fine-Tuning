"""
Master Entry Point: IDE Antigravity LLM Fine-Tuning Pipeline (QLoRA)
"""

import sys
import os

from src.config import QLoRAConfig
from src.hardware_utils import setup_environment_and_seed, print_hardware_summary
from src.dataset_prep import prepare_antigravity_dataset, RAW_ANTIGRAVITY_DATASET
from src.model_loader import load_model_and_tokenizer
from src.trainer import create_sft_trainer, run_training_loop
from src.evaluation import (
    compute_validation_perplexity,
    run_comparative_inference,
    calculate_alignment_metrics
)
from src.export_utils import save_lora_adapters_locally, mount_google_drive_and_export


def main():
    # 1. Environment Setup & Hardware Check
    config = QLoRAConfig()
    hw_info = setup_environment_and_seed(seed=config.random_state)
    print_hardware_summary(hw_info)
    
    # 2. Base Model & Tokenizer Initialization
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 3. Dataset Creation & Chat Formatting
    print("[Pipeline] Preparing IDE Antigravity training & validation dataset...")
    dataset_dict = prepare_antigravity_dataset(tokenizer=tokenizer, val_split_ratio=0.2)
    print(f"[Pipeline] Train samples: {len(dataset_dict['train'])}, Validation samples: {len(dataset_dict['validation'])}")
    
    # 4. Supervised Fine-Tuning (SFT)
    trainer = None
    if model is not None and tokenizer is not None:
        trainer = create_sft_trainer(model, tokenizer, dataset_dict, config)
        run_training_loop(trainer)
    else:
        print("[Pipeline] Running in environment validation mode (Skipping full GPU training loop).")
        
    # 5. Evaluation, Accuracy & Qualitative Testing
    eval_stats = compute_validation_perplexity(trainer, dataset_dict["validation"])
    
    test_prompt = (
        "Diagnose and fix the module import error reported in the terminal log.\n\n"
        "Terminal Output:\n"
        "Traceback (most recent call last):\n"
        "  File 'main.py', line 2, in <module>\n"
        "    import yaml\n"
        "ModuleNotFoundError: No module named 'yaml'\n\n"
        "File: requirements.txt\n"
        "requests>=2.31.0\n"
    )
    
    qual_res = run_comparative_inference(model, tokenizer, prompt=test_prompt)
    
    # Calculate ROUGE/BLEU scores against dataset ground truth targets
    predictions = [qual_res["after_output"]]
    references = [RAW_ANTIGRAVITY_DATASET[1]["output"]]
    metrics = calculate_alignment_metrics(predictions, references)
    
    # 6. Model Export & Saving
    local_saved_path = save_lora_adapters_locally(model, tokenizer, config)
    mount_google_drive_and_export(local_saved_path, config)
    
    print("\n" + "=" * 60)
    print("  IDE ANTIGRAVITY QLORA FINE-TUNING PIPELINE COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
