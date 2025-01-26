import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class PeftLlama:
    PEFT_MODEL_PATHS = {
        "base": "checkpoint/Llama-2-7b-hf",
        "direct": "checkpoint/llama-2-7b-lora-direct",
        "intent": "checkpoint/llama-2-7b-lora-intent",
    }

    def __init__(self, defense_prompt="direct", device="cuda") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.PEFT_MODEL_PATHS['base'])
        base_model = AutoModelForCausalLM.from_pretrained(self.PEFT_MODEL_PATHS['base'], device_map='auto')
        self.model = PeftModel.from_pretrained(base_model, self.PEFT_MODEL_PATHS[defense_prompt])
        # self.model.to(device)
        self.model.eval()

        self.device = device

    def __call__(self, msg):
        msg = msg.replace('</s>', '')
        inputs = self.tokenizer(msg, return_tensors='pt')
        with torch.no_grad():
            generate_ids = self.model.generate(inputs['input_ids'].to(self.device), max_new_tokens=128)
            out = self.tokenizer.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return out
