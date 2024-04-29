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

    def create_prompt(self, info, prev_info):
        print(info, prev_info)
        player_coords = (info['player_x'],info['player_y'])
        enemy_coords = (info['enemy_x'],info['enemy_y'])
        ball_pos = (info['ball_x'],info['ball_y'])
        prev_ball_pos = (prev_info['ball_x'],prev_info['ball_y'])
        output = f'My coordinates are {player_coords} and my enemy\'s coordinates are {enemy_coords}. The ball is at the position {ball_pos}. Right before this, the ball was at {prev_ball_pos}. Following are the six actions available in the format (action id: action) - 0: do nothing, 1: FIRE, 2: RIGHT,  3:LEFT,4:RIGHTFIRE,5:LEFTFIRE. Which action would be the best to take in this situation to win? Please give me the answer just in the format of (action: <action ID>)'
        return output
    # obs to natural language
    def RL2LLM(self, info, prev_info):
        key = str(info)+"#"+str(prev_info)
        if key in self.saved_teacher_recommendations:
            return self.saved_teacher_recommendations[key],False
        context = self.create_prompt(info, prev_info)
        #context = f'You are currently at {state}.\nMove up will reach {up}, move down will reach {down}, move left will reach {left}, move right will reach {right}.\nWhat\'s the best action you should take? Please give me the answer just in the format of (action: <action ID>).'
        return context,True

    def return_action(self,response):
        print("teacher returned - ", response)
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
        for x,y in states:
            text,need_to_prompt = self.RL2LLM(x,y)
            #print(f"Return value from RL2LLM - {text,need_to_prompt}")
            if need_to_prompt:
                action = self.query_codex(text)
                key = str(x)+"#"+str(y)
                if key not in self.saved_teacher_recommendations:
                    self.saved_teacher_recommendations[key] = action
                #print(f"Teacher LLM returned: plans - {plans}")
            else:
                action = text
            #plans[state]=action
        #print(plans)
        return action#self.saved_teacher_recommendations
    