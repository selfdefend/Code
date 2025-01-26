import os
import json
import copy
import argparse

from utils.jailbreak_loader import load_jailbreaker
from utils.gpt import ChatGPT
from model.claude import Claude
from model.hugging_face_llm import HuggingFaceLLM
from utils.checker import JailbreakChecker
from model.peft_model import PeftLlama
from defense.self_defend import SelfDefend


parser = argparse.ArgumentParser()
# jailbreaking setting
parser.add_argument('--jailbreak-path', dest='jailbreak_path', default='./results/manual/DAN_gpt-4-0125-preview.json', help='Json file of the jailbreak prompts')
# parser.add_argument('--target_model', dest='target_model', default=None, choices=['gpt-3.5-turbo-0125', 'gpt-4-0125-preview'])
parser.add_argument('--chat-hyperparameters', dest='chat_hparams', type=str, default='temperature=0,n=1,max_tokens=128,top_p=0.0', help='Parameters of the target model')
parser.add_argument('--add-system-prompt', dest='add_system_prompt', default=False)
parser.add_argument('--check-method', dest='check_method', default='key_words', choices=['key_words', 'gpt', 'multilingual'], help='The methods of cheking if the jailbreak succeed')
# defense setting
parser.add_argument('--defense-method', dest='defense_method', type=str, default='None', choices=['None', 'SelfDefend-basic', 'SelfDefend-tuning'], help='Jailbreak defense methods')
parser.add_argument('--defense-prompt', dest='defense_prompt', default='direct', choices=['direct', 'intent'], help='Choose a defense prompt')
parser.add_argument('--output-dir', dest='out_dir', default='./results/defense', help='Folder to save results')
args = parser.parse_args()


jb_method_target = os.path.splitext(os.path.basename(args.jailbreak_path))[0]
jb_method = jb_method_target.split('_')[0]
target_model = jb_method_target.split('_')[-1]
defense_method = args.defense_method

jailbreaker_data = load_jailbreaker(args.jailbreak_path)

configs = {}
for config_string in args.chat_hparams.split(','):
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

if 'gpt' in target_model:
    chatbot = ChatGPT(model_name=target_model, configs=configs)
elif 'claude' in target_model:
    configs.pop('n', None)
    chatbot = Claude(model_name=target_model, configs=configs)
else:
    chatbot = HuggingFaceLLM(model_name=target_model)

jailbreak_checker = JailbreakChecker(target_model)

defense_checker = None
llm_pre = False
if defense_method == 'None':
    defense_checker = None
elif defense_method == 'SelfDefend-basic':
    defense_checker = SelfDefend(chatbot, args.defense_prompt)
    defense_method = f'{defense_method}-{args.defense_prompt}'
    llm_pre = True
elif defense_method == 'SelfDefend-tuning':
    defense_model = PeftLlama(args.defense_prompt)
    defense_checker = SelfDefend(defense_model, defense_method, args.defense_prompt)
    defense_method = f'{defense_method}-{args.defense_prompt}'
    llm_pre = True
else:
    raise NotImplementedError(f"{defense_method} has not been implemented yet.")

asc_j = 0
asc_d = 0
asc_c = 0
total_response_delays = 0
total_defense_delays = 0
total_stack_delays = 0
records = {"evaluation": []}
for i, jailbreak in enumerate(jailbreaker_data):
    jailbreak_prompt = jailbreak['jailbreak prompt']
    response = jailbreak['target model response']
    response_delay = jailbreak['response delay']
    total_response_delays += response_delay

    if 'jailbreak LLM' in jailbreak:
        pass_j = jailbreak['jailbreak LLM']
    else:
        pass_j = jailbreak_checker.checking_jailbreak(jailbreak_prompt, response, method=args.check_method)
    asc_j += pass_j

    pass_d = 1
    defense_delay = 0
    defense_response = None
    if defense_checker:
        pass_d, defense_delay = defense_checker.defense_checking(jailbreak_prompt)
        defense_response = defense_checker.response_current if llm_pre else None
    asc_d += pass_d
    total_defense_delays += defense_delay

    stack_delay = max(response_delay, defense_delay)
    total_stack_delays += stack_delay

    pass_c = 0
    if pass_j == 1 and pass_d == 1:
        pass_c = 1
        asc_c += 1

    record = copy.deepcopy(jailbreak)
    if defense_response:
        record['defense response'] = defense_response
    record['jailbreak LLM'] = pass_j
    record['pass defense'] = pass_d
    record['attack success'] = pass_c
    record['defense delay'] = defense_delay
    record['stack delay'] = stack_delay
    records['evaluation'].append(record)

    print("[{:3d}/{:3d}], [Jailbreak LLM: {:3d}], [Pass {}: {:3d}], [Total: {:3d}]"
          .format(i+1, len(jailbreaker_data), asc_j, defense_method, asc_d, asc_c))


sub_dir = f'{target_model}/{defense_method}'
output_file_name = f'{jb_method_target}_{defense_method}.json'
output_file = os.path.join(args.out_dir, sub_dir, output_file_name)

records['source file'] = args.jailbreak_path
records['jailbreak method'] = jb_method
records['target model'] = target_model
records['defense method'] = defense_method
records['output file'] = output_file
records['jailbreak success rate'] = asc_j/len(jailbreaker_data)
records['pass defense rate'] = asc_d/len(jailbreaker_data)
records['attack success rate'] = asc_c/len(jailbreaker_data)
records['average response delay'] = total_response_delays/len(jailbreaker_data)
records['average defense delay'] = total_defense_delays/len(jailbreaker_data)
records['average stack delay'] = total_stack_delays/len(jailbreaker_data)

print('Jailbreak: ', jb_method_target)
print('Target Model: ', target_model)
print('Defense Method: ', defense_method)
print("All records are saved to ", output_file)
print('Jailbreak Success Rate: ', asc_j/len(jailbreaker_data))
print('Pass Defense Rate: ', asc_d/len(jailbreaker_data))
print('Attack Success Rate: ', asc_c/len(jailbreaker_data))
print('Average Response Delay: ', total_response_delays/len(jailbreaker_data))
print('Average Defense Delay: ', total_defense_delays/len(jailbreaker_data))
print('Average Stack Delay: ', records['average stack delay'])

if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent='\t', ensure_ascii=False)
    f.close()
