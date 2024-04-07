import os
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv


from abc import ABC, abstractmethod

class teacher_policy():

    def __init__(self, model, prefix=''):
        self.prompt_prefix = prefix
        self.llm_model = model
        self.saved_teacher_recommendations={}
        load_dotenv(".env")
        self.client = OpenAI(api_key=os.environ["KEY"])

    def find_next(self, state):
        output = []
        c, r = state
        if c-1 < 0 or (c-1 == 1 and r == 1):
            output.append((c, r))
        else:
            output.append((c-1, r))
        if c+1 > 2 or (c+1 == 1 and r == 1):
            output.append((c, r))
        else:
            output.append((c+1, r))
        if r-1 < 0 or (c == 1 and r-1 == 1):
            output.append((c, r))
        else:
            output.append((c, r-1))
        if r+1 > 2 or (c == 1 and r+1 == 1):
            output.append((c, r))
        else:
            output.append((c, r+1))

        return output
    # obs to natural language
    def RL2LLM(self, state):
        if state in self.saved_teacher_recommendations:
            return self.saved_teacher_recommendations[state],True
        up, down, left, right = self.find_next(state)
        context = f'You are currently at {state}.\nMove up will reach {up}, move down will reach {down}, move left will reach {left}, move right will reach {right}.\nCan you decided on the best action, please give me the answer just in the format of (action: <action ID>).'
        return context,False

    def return_action(self,response):
        index = response.find("(action: ")
        action = response[index+9]
        #print(response,action)
        return int(action)
        
    def query_codex(self, prompt_text):
        result=''
        server_error_cnt = 0
        while server_error_cnt < 10:
            try:
                result = self.client.chat.completions.create(
                        model=self.llm_model,
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": self.prompt_prefix + prompt_text
                                    }
                                ]
                            }
                        ],
                        max_tokens=4096
                    ).choices[0].message.content
                break
                    
            except Exception as e:
                server_error_cnt += 1
                print(f"fail to query: {e}")
        return self.return_action(result)
    
    def prompt(self, states):
        plans = {}
        for state in states:
            text,need_to_prompt = self.RL2LLM(state)
            print(f"Return value from RL2LLM - {text}")
            if need_to_prompt:
                action = self.query_codex(text)
                if state not in self.saved_teacher_recommendations:
                    self.saved_teacher_recommendations[state] = action
                print(f"Teacher LLM returned: plans - {plans}")
            else:
                action = text
            plans[state]=action
        return plans
    