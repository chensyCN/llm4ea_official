from typing import List
import json
import sys
import time
import openai
import pickle
from config import Config
from random import shuffle

from tqdm import tqdm

def load_links(file_path):
    with open(file_path, 'r') as f:
        return [(line.strip().split('\t')[0].split('/')[-1], line.strip().split('\t')[1].split('/')[-1]) for line in f]

def get_entity_set(file_path):
    with open(file_path, 'r') as f:
        return {line.strip().split('\t')[i].split('/')[-1] for line in f for i in [0, 2]}

class Annotator:

    def __init__(self, api_key: str, messages: List = None):
        openai.api_key = api_key
        if messages:
            self.messages = messages
        else:
            self.messages = [
                {"role": "system", "content": "You are an expert in data mining and knowledge graph alignment."},
                {"role": "user", "content": "Please help me with the task of aligning entities in a knowledge graph. Currently, I have two versions of the knowledge graph, one is built from DBpedia (a structured database) and the other is extracted from wikipedia. These two knowledge graphs have the same set of entities and relationships. Please assist me in identifying the matching entities between these two knowledge graphs based on their entity names and semantics."}
            ]

    def ask_chat_gpt(self) -> str:
        response = openai.ChatCompletion.create(
            model=Config.gpt_model,
            messages=self.messages
        )
        response_content = response['choices'][0]['message']['content']
        return response_content

    def train(self, x1: str, y: str):
        self.messages.append({"role": "user", "content": f"'{x1}' is ture or false?"})
        response_content = self.ask_chat_gpt()
        self.messages.append({"role": "assistant", "content": response_content})

        if 'True.' in response_content:
            if 'True.' == y:
                feedback = "Good job!"
            else:
                feedback = f"You answered incorrectly. You answered '{response_content}', but the correct answer is '{y}'."
        elif 'False.' in response_content:
            if 'False.' == y:
                feedback = "Good job!"
            else:
                feedback = f"You answered incorrectly. You answered '{response_content}', but the correct answer is '{y}'."
        elif 'Not sure.' in response_content:
            if 'Not sure.' == y:
                feedback = "Good job!"
            else:
                feedback = f"You answered incorrectly. You answered '{response_content}', but the correct answer is '{y}'."
        else:
            feedback = "You're answering in the wrong format. If the information described by this entity pair is correct, please reply with 'True'. If not, please reply with 'False' and the corresponding judgment basis. If you are unsure, please reply with 'not sure'. Do not reply with any extra words or punctuation."
        self.messages.append({"role": "user", "content": feedback})

        print(f"\nCurrent training sample: x1={x1}, y={y}")
        print(f"self.messages=")
        for msg in self.messages:
            print(msg)

    def predict(self, x1, x2):
        self.messages.append({"role": "user", "content": f"Given an entity '{x1}' in the English knowledge graph, please help me determine which of the following entities in '{x2}' corresponds to '{x1}' in the German knowledge graph. Please directly reply with the name of the target entity. Do not reply with any extra words or punctuation."})
        response_content = self.ask_chat_gpt()
        self.messages.pop()
        return response_content

    def save(self, model_path: str):
        model_dict = {
            'messages': self.messages
        }
        with open(model_path, "w", encoding='utf-8') as f:
            json.dump(model_dict, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(self, model_path: str, api_key: str) -> 'Annotator':
        with open(model_path, "r", encoding='utf-8') as f:
            model_dict = json.load(f)
            model = Annotator(api_key=api_key, messages=model_dict['messages'])
            return model
