from model.vllmModel import vllmModel
from evaluator.Evaluator import MedMCQAEvaluator  

def main():
    vm = vllmModel(
        model_name="LLaMA-Factory/saves/llama3-8b/lora/sft/model",
        gpu_ids="0",
        max_tokens=64,
        temperature=0.7,
    )

    ev = MedMCQAEvaluator(vm, batch_size=512)

    metrics = ev.evaluate(
        test_path="data/medmcqa/processed/dev.jsonl",
        out_pred_path="./outputs/medmcqa/llama8b+lora/predictions.jsonl",
        out_metrics_path="./outputs/medmcqa/llama8b+lora/metrics.json",
        max_examples=None, 
        show_progress=True,
    )

    print(metrics)

if __name__ == "__main__":
    main()
