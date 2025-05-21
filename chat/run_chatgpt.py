import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# Set API key and base URL
# api_key = "606ff88a999344ff9517880554752464"
# api_base = "https://api.aimlapi.com/v1"

api_key = ""
api_base = ""

# Read the text
folder = "GPT4o_result_GenWiki-Hard"
with open(f'./datasets/{folder}/test.target', 'r') as f:
    text = [l.strip() for l in f.readlines()]

async def api_model(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    response = await openai_async_client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0, **kwargs
    )

    return response.choices[0].message.content

async def _run_api(queries, max_concurrent=16):
    semaphore = asyncio.Semaphore(max_concurrent)  # Limit maximum concurrency to 8

    async def limited_api_model(query):
        async with semaphore:
            return await api_model(query)

    tasks = [limited_api_model(query) for query in queries]
    answers = await tqdm.gather(*tasks)
    return answers

async def process_texts():
    queries = []
    for t in text:
        prompt = (
            f"""
Transform the text into a semantic graph.

Example#1:
Text: Shotgate \"Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.\"
Semantic Graph: 
```[["Shotgate Thickets", "instance of", "Nature reserve"], ["Shotgate Thickets", "country", "United Kingdom"], ["Shotgate Thickets", "operator", "Essex Wildlife Trust"]]```
Example#2:
Text: The Eiffel Tower, located in Paris, France, is a famous landmark and a popular tourist attraction. It was designed by the engineer Gustave Eiffel and completed in 1889.
Semantic Graph:
```[["Eiffel Tower", "located in", "Paris"], ["Eiffel Tower", "located in", "France"], ["Eiffel Tower", "instance of", "landmark"], ["Eiffel Tower", "attraction type", "tourist attraction"],["Eiffel Tower", "designed by", "Gustave Eiffel"], ["Eiffel Tower", "completion year", "1889"]]```

Note: 
1. Make sure each item in the list is a triple with strictly three items.
2. Give me the answer in the form of a semantic graph extactly as example shown, which is a list of lists.
3. DO NOT output like '```json[ .. ]```', SHOUDLE BE LIKE '```[...]```'.

Text: \"{t}\"
Semantic Graph:
"""
        )
        queries.append(prompt)

    responses = await _run_api(queries)

    with open(f"./datasets/{folder}/gpt_baseline/test_generated_graphs.txt", "w") as output_file:
        for response in responses:
            output_file.write(response.strip().replace('\n', '') + '\n')

# Run the async process
asyncio.run(process_texts())