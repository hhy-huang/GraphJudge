import os
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# 设置 API 密钥和基础 URL
api_key = ""
api_base = ""

# 读取待去噪的文本
# dataset = "rebel_sub"
dataset = "GenWiki-Hard"  # rebel / webnlg / kelm
# dataset = "SCIERC"
dataset_path = f'./datasets/GPT4o_mini_result_{dataset}/'
Iteration = 1

if Iteration == 1:  # 原始文本
    with open(dataset_path + 'test.target', 'r') as f:
        text = [l.strip() for l in f.readlines()]
else:  # 上一次迭代的去噪文本
    with open(dataset_path + f'Iteration{Iteration - 1}/test_denoised.target', 'r') as f:
        text = [l.strip() for l in f.readlines()]

async def api_model(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    openai_async_client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    response = await openai_async_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0, **kwargs
    )
    return response.choices[0].message.content

async def _run_api(queries, max_concurrent=8):
    semaphore = asyncio.Semaphore(max_concurrent)  # 限制最大并发数为8

    async def limited_api_model(query):
        async with semaphore:
            return await api_model(query)

    tasks = [limited_api_model(query) for query in queries]
    answers = await tqdm.gather(*tasks)
    return answers

async def extract_entities(texts):
    prompts = []
    for t in texts:
        prompt = f"""
Goal:
Transform the text into a list of entities.

Here are two examples:
Example#1:
Text: "Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust."
List of entities: ["Shotgate Thickets", "Nature reserve", "United Kingdom", "Essex Wildlife Trust"]
Example#2:
Text: "Garczynski Nunatak () is a cone-shaped nunatak, the highest in a cluster of nunataks close west of Mount Brecher, lying at the north flank of Quonset Glacier in the Wisconsin Range of the Horlick Mountains of Antarctica. It was mapped by the United States Geological Survey from surveys and U.S. Navy air photos, 1959–60, and was named by the Advisory Committee on Antarctic Names for Carl J. Garczynski, a meteorologist in the Byrd Station winter party, 1961."
List of entities: ["Garczynski Nunatak", "nunatak", "Wisconsin Range", "Mount Brecher", "Quonset Glacier", "Horlick Mountains"]

Refer to the examples and here is the question:
Text: "{t}"
List of entities: """
        prompts.append(prompt)
    
    entities_list = await _run_api(prompts)
    return entities_list

async def denoise_text(texts, entities_list):
    prompts = []
    for t, entities in zip(texts, entities_list):
        prompt = f"""
Goal:
Denoise the raw text with the given entities, which means remove the unrelated text and make it more formatted.

Here are two examples:
Example#1:
Raw text: "Zakria Rezai (born 29 July 1989) is an Afghan footballer who plays for Ordu Kabul F.C., which is a football club from Afghanistan. He is also an Afghanistan national football team player, and he has 9 caps in the history. He wears number 14 on his jersey and his position on field is centre back."
Entities: ["Zakria Rezai", "footballer", "Ordu Kabul F.C.", "Afghanistan", "29 July 1989"]
Denoised text: "Zakria Rezai is a footballer. Zakria Rezai is a member of the sports team Ordu Kabul F.C. Zakria Rezai has the citizenship of Afghanistan. Zakria Rezai was born on July 29, 1989. Ordu Kabul F.C. is a football club. Ordu Kabul F.C. is based in Afghanistan."
Example#2:
Raw text: "Elizabeth Smith, a renowned British artist, was born on 12 May 1978 in London. She is specialized in watercolor paintings and has exhibited her works in various galleries across the United Kingdom. Her most famous work, 'The Summer Breeze,' was sold at a prestigious auction for a record price. Smith is also a member of the Royal Society of Arts and has received several awards for her contributions to the art world."
Entities: ["Elizabeth Smith", "British artist", "12 May 1978", "London", "watercolor paintings", "United Kingdom", "The Summer Breeze", "Royal Society of Arts"]
Denoised text: "Elizabeth Smith is a British artist. Elizabeth Smith was born on May 12, 1978. Elizabeth Smith was born in London. Elizabeth Smith specializes in watercolor paintings. Elizabeth Smith's artwork has been exhibited in the United Kingdom. 'The Summer Breeze' is a famous work by Elizabeth Smith. Elizabeth Smith is a member of the Royal Society of Arts."

Refer to the examples and here is the question:
Raw text: {t}
Entities: {entities}
Denoised text: """
        prompts.append(prompt)
    
    denoised_texts = await _run_api(prompts)
    return denoised_texts

async def main():
    # 提取实体并保存
    entities_list = await extract_entities(text)
    with open(dataset_path + f"Iteration{Iteration}/test_entity.txt", "w") as output_file:
        for entities in entities_list:
            output_file.write(entities.strip().replace('\n', '') + '\n')
    
    # 读取提取的实体
    last_extracted_entities = []
    with open(dataset_path + f'Iteration{Iteration}/test_entity.txt', 'r') as f:
        for l in f.readlines():
            last_extracted_entities.append(l.strip())
    
    # 去噪文本并保存
    denoised_texts = await denoise_text(text, last_extracted_entities)
    with open(dataset_path + f"Iteration{Iteration}/test_denoised.target", "w") as output_file:
        for denoised_text in denoised_texts:
            output_file.write(denoised_text.strip().replace('\n', '') + '\n')

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())