import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import torch
import argparse
import pandas as pd
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset
import transformers
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# model config
BASE_MODEL = "NousResearch/Llama-2-7b-hf"
# LORA_WEIGHTS = "models/llama2-7b-lora-wn18rr/"
# LORA_WEIGHTS = "models/llama2-7b-lora-wn11/"
# LORA_WEIGHTS = "models/llama2-7b-lora-FB13/"

# LORA_WEIGHTS = "models/llama2-7b-lora-rebel-sub-context-tmp/"
LORA_WEIGHTS = "models/llama2-7b-lora-scierc-context-0518"
# LORA_WEIGHTS = "models/llama2-7b-lora-genwiki-context-tmp/"
# LORA_WEIGHTS = "models/llama2-7b-lora-genwiki-20250508/" 

# tokenizer
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
LOAD_8BIT = False

# device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

# model
model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        # torch_dtype=torch.float16,
        # device_map="auto",
    ).half().cuda()
model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        # torch_dtype=torch.float16,
    ).half().cuda()
# if not LOAD_8BIT:
#     model.half()  # seems to fix bugs for some users.

def generate_prompt(instruction):
    prompt = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:
"""
    return prompt

def generate_advanced_prompt(instruction, input=None):
    prompt = f"""
Goal:
You need to do the graph judgement task, which means you need to clarify
 the correctness of the given triple.
Attention:
1.The correct triple sentence should have a correct grammatical structure.
2.The knowledge included in the triple sentence should not conflict with
the knowledge you have learned.
3.The answer should be either "Yes, it is true." or "No, it is not true."

Here are two examples:
Example#1:
Question: Is this ture: Apple Founded by Mark Zuckerberg ?
Answer: No, it is not true.
Example#2:
Question: Is this true: Mark Zuckerberg Founded Facebook ?
Answer: Yes, it is true.

Refer to the examples and here is the question:
Question: {instruction}
Answer:
"""
    return prompt

model.eval()

def evaluate(
    instruction,
    input=None,
    temperature=0,       # 0.1
    top_p=0.75,
    top_k=10,
    num_beams=4,
    max_new_tokens=64,      # 256
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    print("Input:")
    print(prompt)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print("Output:")
    print(output)
    
    return output.split("### Response:")[1].strip()

def batch_evaluate(
    instructions,
    temperature=0,       # 0.1
    top_p=0.75,
    top_k=10,
    num_beams=4,
    max_new_tokens=64,      # 256
    **kwargs,
):
    prompts = [generate_prompt(inst) for inst in instructions]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # 你可以根据实际情况调整
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,  # 新增
        )
    outputs = []
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        if "### Response:" in output:
            outputs.append(output.split("### Response:")[1].strip())
        else:
            outputs.append(output)
    return outputs

if __name__ == "__main__":
    # testing code for readme
    parser = argparse.ArgumentParser()
    # parser.add_argument("--finput", type=str, default="data/WN11/test_instructions_llama.csv")
    # parser.add_argument("--foutput", type=str, default="data/WN11/pred_instructions_llama2_7b.csv")
    # parser.add_argument("--finput", type=str, default="data/FB13/test_instructions_llama.csv")
    # parser.add_argument("--foutput", type=str, default="data/FB13/pred_instructions_llama2_7b.csv")
    parser.add_argument("--finput", type=str, default="data/scierc_4omini_context/test_instructions_context_llama2_7b.json")
    parser.add_argument("--foutput", type=str, default="data/scierc_4omini_context/pred_instructions_context_llama2_7b_gj.csv")  # pred_instructions_context_genwiki_llama2_7b_itr1.csv
    # parser.add_argument("--finput", type=str, default="data/WN18RR/test_instructions_llama_merge.csv")
    # parser.add_argument("--foutput", type=str, default="data/WN18RR/pred_instructions_llama2_7b_merge.csv")
    args = parser.parse_args()

    total_input = load_dataset("json", data_files=args.finput)
    data_eval = total_input["train"]

    instruct, pred = [], []
    total_num = len(data_eval)
    batch_size = 16  # 可根据显存调整
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
