#!/usr/bin/env python3
"""
Medical Model Evaluation Script
Evaluates both base model and fine-tuned models on medical reasoning tasks
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

os.environ["TRANSFORMERS_VERBOSITY"] = "info"

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer"""
    # Handle local paths
    if os.path.exists(model_path):
        model_path = os.path.abspath(model_path)
        print(f"Loading local model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        print(f"ERROR loading model from {model_path}: {e}")
        exit(1)

def format_medical_question(example):
    """Format medmcqa question"""
    question = example['question']
    choices = [
        f"A) {example['opa']}",
        f"B) {example['opb']}",
        f"C) {example['opc']}",
        f"D) {example['opd']}"
    ]
    
    formatted = f"{question}\n" + "\n".join(choices) + "\n\nAnswer:"
    return formatted

def extract_answer(generated_text):
    """Extract the multiple choice answer (A, B, C, D) from model response.
    Prioritizes "The answer is X" pattern for consistency with training."""
    import re
    
    # Method 1: Look for "The answer is X" patterns (HIGHEST PRIORITY)
    answer_match = re.search(r'(?:the answer is|answer is|answer:|is)\s*([ABCD])', generated_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Method 2: Look for answer in <answer> tags with "The answer is X" format
    answer_match = re.search(r'<answer>\s*(?:the answer is|answer is)?\s*([ABCD])\s*</answer>', generated_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Method 3: Look for answer in <answer> tags (any format)
    answer_match = re.search(r'<answer>\s*([ABCD])\s*</answer>', generated_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Method 4: Look for first letter at start of response
    answer_match = re.search(r'^\s*([ABCD])\b', generated_text.strip(), re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Method 5: Look for standalone letters at end of response  
    answer_match = re.search(r'\b([ABCD])\b(?!.*\b[ABCD]\b)', generated_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Method 6: Look for any A, B, C, D in the response (last resort)
    letters = re.findall(r'\b([ABCD])\b', generated_text, re.IGNORECASE)
    if letters:
        return letters[0].upper()  # Return the FIRST found letter
    
    # Default: return None if no answer found
    return None

def evaluate_model_on_medmcqa(model, tokenizer, dataset, num_samples=1000, debug=True):
    """Evaluate model on medmcqa dataset"""
    correct = 0
    total = 0
    debug_count = 0
    no_answer_count = 0
    cop_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    print(f"Evaluating on {min(num_samples, len(dataset))} medmcqa samples...")
    
    for i, example in enumerate(tqdm(dataset)):
        if i >= num_samples:
            break
            
        # Format question
        prompt = format_medical_question(example)
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract generated text
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Improved answer extraction
        predicted = extract_answer(generated)
        
        # Get correct answer using 0-based indexing (same as rewards.py)
        cop_value = example.get('cop', -1)
        correct_answer = cop_to_letter.get(cop_value, None)
        
        # Count cases where no answer was extracted
        if predicted is None:
            no_answer_count += 1
            predicted = "A"  # Default fallback for scoring
        
        # Debug: Show first few examples
        if debug and debug_count < 5:
            print(f"\n--- DEBUG EXAMPLE {debug_count + 1} ---")
            print(f"Question: {example['question'][:100]}...")
            print(f"Generated: '{generated}'")  # Show full generated text
            
            # Debug the extraction process step by step
            import re
            print(f"Raw generated text: '{generated[:100]}...'")
            
            # Test each method individually
            method1 = re.search(r'(?:the answer is|answer is|answer:|is)\s*([ABCD])', generated, re.IGNORECASE)
            method2 = re.search(r'<answer>\s*(?:the answer is|answer is)?\s*([ABCD])\s*</answer>', generated, re.IGNORECASE)
            method3 = re.search(r'<answer>\s*([ABCD])\s*</answer>', generated, re.IGNORECASE)
            method4 = re.search(r'^\s*([ABCD])\b', generated.strip(), re.IGNORECASE)
            method5 = re.search(r'\b([ABCD])\b(?!.*\b[ABCD]\b)', generated, re.IGNORECASE)
            all_letters = re.findall(r'\b([ABCD])\b', generated, re.IGNORECASE)
            
            print(f"Method 1 - 'The answer is' pattern: {method1.group(1).upper() if method1 else None}")
            print(f"Method 2 - Answer tag with 'is' pattern: {method2.group(1).upper() if method2 else None}")
            print(f"Method 3 - Answer tag match: {method3.group(1).upper() if method3 else None}")
            print(f"Method 4 - First letter match: {method4.group(1).upper() if method4 else None}")
            print(f"Method 5 - Last standalone letter: {method5.group(1).upper() if method5 else None}")
            print(f"Method 6 - All letters found: {[l.upper() for l in all_letters]} (first={all_letters[0].upper() if all_letters else None})")
            print(f"FINAL Predicted: {predicted}")
            print(f"Correct: {correct_answer} (cop={cop_value})")
            print(f"Match: {predicted == correct_answer}")
            debug_count += 1
        
        if predicted == correct_answer:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n📊 Evaluation Statistics:")
    print(f"  • No answer extracted: {no_answer_count}/{total} ({no_answer_count/total*100:.1f}%)")
    print(f"  • Answer extraction success: {(total-no_answer_count)/total*100:.1f}%")
    
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate medical reasoning models")
    parser.add_argument("--base_model", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="Base model to evaluate")
    parser.add_argument("--finetuned_model", type=str,
                       help="Path to fine-tuned model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to evaluate (for faster testing)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🏥 Medical Model Evaluation Pipeline")
    print("=" * 50)
    
    # Load medmcqa validation dataset
    print("Loading medmcqa validation dataset...")
    dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
    
    results = {}
    
    # Evaluate base model
    print(f"\n📊 Evaluating BASE MODEL: {args.base_model}")
    print("=" * 60)
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model)
    base_acc, base_correct, base_total = evaluate_model_on_medmcqa(
        base_model, base_tokenizer, dataset, args.num_samples
    )
    
    results['base_model'] = {
        'model_path': args.base_model,
        'accuracy': base_acc,
        'correct': base_correct,
        'total': base_total,
        'percentage': f"{base_acc * 100:.2f}%"
    }
    
    print(f"Base Model Results: {base_correct}/{base_total} = {base_acc * 100:.2f}%")
    
    # Clean up base model from memory
    del base_model, base_tokenizer
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model if provided
    if args.finetuned_model:
        print(f"\n🎯 Evaluating FINE-TUNED MODEL: {args.finetuned_model}")
        print("=" * 60)
        ft_model, ft_tokenizer = load_model_and_tokenizer(args.finetuned_model)
        ft_acc, ft_correct, ft_total = evaluate_model_on_medmcqa(
            ft_model, ft_tokenizer, dataset, args.num_samples
        )
        
        results['finetuned_model'] = {
            'model_path': args.finetuned_model,
            'accuracy': ft_acc,
            'correct': ft_correct,
            'total': ft_total,
            'percentage': f"{ft_acc * 100:.2f}%"
        }
        
        print(f"Fine-tuned Model Results: {ft_correct}/{ft_total} = {ft_acc * 100:.2f}%")
        
        # Calculate improvement
        improvement = ft_acc - base_acc
        results['improvement'] = {
            'absolute': improvement,
            'relative': f"{improvement * 100:.2f} percentage points",
            'better': improvement > 0
        }
        
        print(f"\n📈 IMPROVEMENT: {improvement * 100:.2f} percentage points")
        if improvement > 0:
            print("✅ Fine-tuning improved performance!")
        else:
            print("❌ Fine-tuning did not improve performance")
    
    # Save results
    output_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n✅ Evaluation complete! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
