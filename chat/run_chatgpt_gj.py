import os
import aiohttp
import asyncio
import json
import csv
from tqdm.asyncio import tqdm
from datasets import load_dataset

# Set API key and base URL
api_key = ""
api_base = ""


folder = "GPT4o_mini_result_rebel_sub"
# Input and output file paths
input_file = f"./datasets/{folder}/test_instructions_context_llama2_7b.json"
output_file = f"./datasets/{folder}/pred_instructions_context_llama2_7b_gpt_mini.csv"

# Load instructions from JSON file
total_input = load_dataset("json", data_files=input_file)
data_eval = total_input["train"].train_test_split(
    test_size=499, shuffle=True, seed=42
)["test"]

with open(input_file, "r", encoding="utf-8") as f:
    instructions = json.load(f)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

async def get_chatgpt_completion(session, instruction, input_text):
    """
    Send prompt to GPT and get the generated response.
    """
    url = f"{api_base}/chat/completions"
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    while True:
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                # Return the first completion result
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            await asyncio.sleep(1)

async def process_instructions():
    """
    Process each instruction and generate responses using GPT.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for item in data_eval:
            instruction = item["instruction"]
            input_text = item["input"]
            tasks.append(get_chatgpt_completion(session, instruction, input_text))

        # Execute all tasks and gather responses
        responses = await tqdm.gather(*tasks)

        # Write responses to a CSV file
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "generated"])  # Write header

            for item, response in zip(instructions, responses):
                prompt = item["instruction"]
                writer.writerow([prompt, response.strip()])

# Run the async process
asyncio.run(process_instructions())