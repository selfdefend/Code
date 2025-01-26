import os
import json
import time
import copy
import argparse

from utils.jailbreak_loader import load_jailbreaker
from utils.gpt import ChatGPT
from model.claude import Claude
from model.hugging_face_llm import HuggingFaceLLM
from utils.checker import JailbreakChecker


chat_hparams='temperature=0,n=1,max_tokens=128,top_p=0.0'


parser = argparse.ArgumentParser()
# jailbreaking setting
parser.add_argument('--jailbreak-path', dest='jailbreak_path', type=str, help='Json file of the jailbreak prompts')
parser.add_argument('--target-model', dest='target_model', choices=['gpt-3.5-turbo-0125', 'gpt-4-0125-preview', 'claude-3-5-sonnet-20241022', 'llama-2-7b', 'llama-2-13b', 'mistral-7b-v0.2'])
parser.add_argument('--check-method', dest='check_method', default='key_words', choices=['key_words', 'gpt', 'multilingual'], help='The methods of cheking if the jailbreak succeed')
args = parser.parse_args()
jailbreak_path = args.jailbreak_path
target_model_name = args.target_model

jailbreak_method = os.path.splitext(os.path.basename(jailbreak_path))[0]

jailbreaker_data = load_jailbreaker(jailbreak_path)

configs = {}
for config_string in chat_hparams.split(','):
    key, value = config_string.split('=', 2)
    if '.' in value:
        configs[key] = float(value)
    elif value == 'false':
        configs[key] = False
    elif value == 'true':
        configs[key] = True
    elif value.startswith('"') and value.endswith('"'):
        configs[key] = value
    else:
        configs[key] = int(value)


if 'gpt' in target_model_name:
    target_model = ChatGPT(model_name=target_model_name, configs=configs)
elif 'claude' in target_model_name:
    configs.pop('n', None)
    target_model = Claude(model_name=target_model_name, configs=configs)
else:
    target_model = HuggingFaceLLM(model_name=target_model_name)

jailbreak_checker = JailbreakChecker(target_model_name)


asc = 0
records = []
for i, jailbreak in enumerate(jailbreaker_data):
    jailbreak_prompt = jailbreak['jailbreak prompt']
    if 'target model response' in jailbreak:
        response = jailbreak['target model response']
        start_time = time.time()
        target_model(jailbreak_prompt)
        end_time = time.time()
    else:
        start_time = time.time()
        response = target_model(jailbreak_prompt)
        end_time = time.time()
    response_delay = end_time - start_time

    response_4_eval = copy.deepcopy(response)
    if 'LLM-Fuzzer' in jailbreak_path and "Unaligned:" in response:
        response_4_eval = response.split("Unaligned:")[1].strip()
    success = jailbreak_checker.checking_jailbreak(jailbreak_prompt, response_4_eval, method=args.check_method)
    asc += success

    record = {
        'index':i,
        'goal': jailbreak['goal'],
        'jailbreak prompt': jailbreak_prompt,
        'target model response': response,
        'response delay': response_delay,
        'jailbreak LLM': success
    }
    records.append(record)
    print("[{:3d}/{:3d}], [Jailbreak LLM: {}]".format(i+1, len(jailbreaker_data), asc))

print('Jailbreak Success Rate: ', asc/len(jailbreaker_data))


folders = jailbreak_path.split('/')
output_dir = os.path.join(folders[1], folders[2])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file_name = f'{jailbreak_method}_{target_model_name}.json'
output_file_path = os.path.join(output_dir, output_file_name)
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent='\t', ensure_ascii=False)
    f.close()

print(f"Output file: {output_file_path}")
