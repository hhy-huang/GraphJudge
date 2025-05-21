import os
import asyncio
import json
import functools
import ast
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# Set API key and base URL

api_key = ""
api_base = ""
openai_async_client = AsyncOpenAI(api_key=api_key, base_url=api_base)

# Read the text to be denoised
text = []
entity = []
# dataset = "rebel_sub"
dataset = "GenWiki-Hard"
# dataset = "SCIERC"
dataset_path = f'./datasets/GPT4o_mini_result_{dataset}/'
Denoised_Iteration = 1
Graph_Iteration = 1

# Read denoised text
with open(dataset_path + f'Iteration{Denoised_Iteration}/test_denoised.target', 'r') as f:
    text = [l.strip() for l in f.readlines()]

# Read the corresponding entities
with open(dataset_path + f'Iteration{Denoised_Iteration}/test_entity.txt', 'r') as f:
    entity = [l.strip() for l in f.readlines()]

async def api_model(prompt, **kwargs):
    messages = [{"role": "user", "content": prompt}]
    response = await openai_async_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.1, **kwargs
    )
    return response.choices[0].message.content

async def _run_api(prompts, max_concurrent=8):
    semaphore = asyncio.Semaphore(max_concurrent)
    async def limited_api_model(prompt):
        async with semaphore:
            return await api_model(prompt)
    tasks = [limited_api_model(prompt) for prompt in prompts]
    answers = await tqdm.gather(*tasks)
    return answers

async def main():
    prompts = []
    for i in range(len(text)):
        prompt = (
                f"Goal:\nTransform the text into a semantic graph(a list of triples) with the given text and entities. "
                f"In other words, You need to find relations between the given entities with the given text.\n"
                f"Attention:\n1.Generate triples as many as possible. "
                f"2.Make sure each item in the list is a triple with strictly three items.\n\n"
                f"Here are two examples:\n"
                f"Example#1: \nText: \"Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.\"\n"
                f"Entity List: [\"Shotgate Thickets\", \"Nature reserve\", \"United Kingdom\", \"Essex Wildlife Trust\"]\n"
                f"Semantic Graph: [[\"Shotgate Thickets\", \"instance of\", \"Nature reserve\"], "
                f"[\"Shotgate Thickets\", \"country\", \"United Kingdom\"], [\"Shotgate Thickets\", \"operator\", \"Essex Wildlife Trust\"]]\n"
                f"Example#2:\nText: \"The Eiffel Tower, located in Paris, France, is a famous landmark and a popular tourist attraction. "
                f"It was designed by the engineer Gustave Eiffel and completed in 1889.\"\n"
                f"Entity List: [\"Eiffel Tower\", \"Paris\", \"France\", \"landmark\", \"Gustave Eiffel\", \"1889\"]\n"
                f"Semantic Graph: [[\"Eiffel Tower\", \"located in\", \"Paris\"], [\"Eiffel Tower\", \"located in\", \"France\"], "
                f"[\"Eiffel Tower\", \"instance of\", \"landmark\"], [\"Eiffel Tower\", \"attraction type\", \"tourist attraction\"], "
                f"[\"Eiffel Tower\", \"designed by\", \"Gustave Eiffel\"], [\"Eiffel Tower\", \"completion year\", \"1889\"]]\n\n"
                f"Refer to the examples and here is the question:\nText: {text[i]}\nEntity List:{entity[i]}\nSemantic graph:"
            )
        prompts.append(prompt)

    responses = await _run_api(prompts)

    # 写入文件
    with open(dataset_path + f"Graph_Iteration{Graph_Iteration}/test_generated_graphs.txt", "w") as output_file:
        for response in responses:
            output_file.write(response.strip().replace('\n', '') + '\n')

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())