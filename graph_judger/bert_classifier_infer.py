import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification

# model config
BASE_MODEL = "google-bert/bert-base-uncased"
#WEIGHTS_PATH = "models/bert-base-classifier-scierc"
WEIGHTS_PATH = "models/bert-base-classifier-rebel-sub"

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained(WEIGHTS_PATH)
model = BertForSequenceClassification.from_pretrained(
    WEIGHTS_PATH,
    num_labels=2
).half().cuda()

def process_input(instruction, input_text=None):
    if input_text:
        text = f"Instruction: {instruction} Input: {input_text}"
    else:
        text = f"Instruction: {instruction}"
    return text

def batch_evaluate(
    instructions,
    max_length=512,
    batch_size=32
):
    # Process all inputs
    processed_texts = [process_input(inst) for inst in instructions]
    
    # Tokenize
    inputs = tokenizer(
        processed_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Prediction
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
    # Get predictions
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_labels = torch.argmax(predictions, dim=-1)
    confidence_scores = torch.max(predictions, dim=-1)[0]
    
    # Convert to readable format
    results = []
    for label, score in zip(predicted_labels, confidence_scores):
        label = label.item()
        score = score.item()
        if label == 1:
            result = "Yes, it is true."
        else:
            result = "No, it is not true."
        results.append(f"{result} (confidence: {score:.3f})")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finput", type=str, default="data/scierc_4omini_context/test_instructions_context_llama2_7b.json")
    parser.add_argument("--foutput", type=str, default="data/scierc_4omini_context/pred_instructions_context_bert_classifier.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    total_input = load_dataset("json", data_files=args.finput)
    data_eval = total_input["train"]

    instruct, pred = [], []
    total_num = len(data_eval)
    batch_size = args.batch_size  # 可根据显存调整
    prompts = data_eval['instruction']
    for i in tqdm(range(0, total_num, batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        tqdm.write(f'{i}/{total_num}')
        batch_responses = batch_evaluate(batch_prompts)
        for cur_instruct, cur_response in zip(batch_prompts, batch_responses):
            cur_response = cur_response.replace('\n', ',')
            pred.append(cur_response)
            instruct.append(cur_instruct)
    output = pd.DataFrame({'prompt': instruct, 'generated': pred})
    output.to_csv(args.foutput, header=True, index=False)