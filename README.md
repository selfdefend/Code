# SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner

This is the source code for our USENIX Security 2025 paper "SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner", which establishes a shadow LLM as a defense instance (in detection state) to concurrently protect the target LLM instance (in normal answering state) in the normal stack and collaborate with it for checkpoint-based access control. In this repository, we not only provide the implementation of the proposed SelfDefend framework, but also how to reproduce its defense results.

## 1. Usage
#### 1.1 Core Environment
- Python 3.11.5
- CUDA 12.4
- Pytorch 2.5.1
- transformers 4.46.3
- peft 0.14.0
- openai 1.55.0
- anthropic 0.21.3

#### 1.2 API Keys
For commercial GPT-3.5/4 and Claude, please go to [gpt.py](./utils/gpt.py) and [claude.py](./model/claude.py) to set their API keys respectively.

#### 1.3 Open-source Models
Because the defense models in our paper are fine-tuned on [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf), please download it from HuggingFace, and put it in the "checkpoint" folder. We have already provided the paired lora parameters in "checkpoint", where "llama-2-7b-lora-direct" is the lora modules for $P_{direct}$-tuned model, and "llama-2-7b-lora-intent" is for $P_{intent}$.

For open-source target models, please download them from HuggingFace and also put them in "checkpoint". The following list shows our supported open-source target LLMs:
- [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [Llama-2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
- [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)


## 2. Dataset
In "./data" folder, we provide datasets used in the paper. Our description of each dataset is as follows:
```
AlpacaEval                          # AlpacaEval dataset
DAN                                 # JailbreakHub dataset
JailbreakBench                      # JailbreakBench benchmark
MultiJail                           # MultiJail dataset
red-team
 ├── red_team_attempts.jsonl        # Anthropic red-team dataset
 ├── red_team_direct.csv            # Direct dataset for $P_{direct}$-tuned model
 ├── red_team_intent.csv            # Intent dataset for $P_{intent}$-tuned model
```


## 3. Starting to Evaluate SelfDefend
1. **Jailbreaking Target Model**

Here, we have provided constructed jailbreak prompts of different jailbreak methods in "attack/results". You can also refer to the existing attack methods to generate jailbreaks for the target model. Then we can run "jailbreaking.py" to compute the ASR of the jailbreak method on the target LLM, e.g.,
```
python jailbreaking.py --jailbreak-path 'attack/results/primary/GCG_individual_llama2-7b.json' --target-model 'llama-2-7b'
```
After running jailbreaking.py, the records will be saved in "results/" as a json file.

2. **Evaluate Defense Performance of SelfDefend**

We use the above saved json file to further evaluate the defense effectiveness of SelfDefend. For non-tuning SelfDefend (i.e., GPT-3.5/4 as a shadow model), we can use the following command to evaluate its defense performance against jailbreaks:
```
python evaluate.py --jailbreak-path ./results/manual/DAN_gpt-3.5-turbo-0125.json --defense-method SelfDefend-basic --defense-prompt direct
```
For tuning-based SelfDefend, we can run this following command:
```
python evaluate.py --jailbreak-path ./results/manual/DAN_gpt-3.5-turbo-0125.json --defense-method SelfDefend-tuning --defense-prompt intent
```
After running "evaluate.py", the defense result of SelfDefend will be saved in "results/defense/" as a json file. In the json file, we record ASRs and delays of our SelfDefend. It is noted that:
```
jailbreak success rate    # ASR for the target LLM
pass defense rate         # ASR for the shadow LLM
attack success rate       # ASR for SelfDefend
average response delay    # average delay of the target LLM
average defense delay     # average delay of the shadow LLM
average stack delay       # average delay of the whole SelfDefend
```


## 4. Fine-tune Defense Models
If you want to fine tune your own defense models, you can make use of "fine_tuning.py", e.g., fine-tuning on Llama-3.1-8B:
```
python fine_tuning.py --base-model llama3.1
```