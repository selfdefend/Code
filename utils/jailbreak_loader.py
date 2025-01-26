import json


def load_jailbreaker(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data
