import os
import time
from anthropic import Anthropic


class Claude:
    def __init__(self, model_name="claude-3-5-sonnet-20241022", configs=None) -> None:
        '''
        model_name: api version of a claude model (https://docs.anthropic.com/en/docs/about-claude/models)
        '''

        self.model_name = model_name
        self.client = Anthropic(
            base_url=os.environ.get("BASE_URL"),
            api_key=os.environ.get("API_KEY")
        )

        self.config = {
                'temperature': 1,
                'top_p': 1,
                'max_tokens': 1024
            }
        if configs is not None:
            self.config = configs
    
    def fit_message(self, msg, system_prompt=None):
        if system_prompt is not None:
            conversation = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": msg}
                    ]
        else:
            conversation = [
                        {"role": "user", "content": msg}
                    ]
        return conversation

    def query(self, msg, system_prompt=None, **kwargs):
        start = time.time()

        while True:
            try:
                raw_response = self.client.messages.create(
                            model=self.model_name,
                            messages=self.fit_message(msg, system_prompt),
                            **kwargs)
                self.raw_response = raw_response

                return raw_response.content[0].text
            except Exception as e:
                print(e)

            current = time.time()
            if current - start > 60: return None
            time.sleep(10)
    
    def __call__(self, msg, system_prompt=None):
        return self.query(msg, system_prompt, **self.config)


# chatbot = Claude()
# print(chatbot("Hello, Claude"))
