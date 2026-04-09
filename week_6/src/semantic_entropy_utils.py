import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import math

import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cluster_assignment_entropy(semantic_ids):
    '''
    Calculates the actual semantic entropy
    '''
    counts = np.bincount(semantic_ids)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs))
    

'''
ENTAILMENT MODEL CHECKS ENTAILMENT
'''
class EntailmentDeberta:
    def __init__(self):
        '''
        initializes the tokenizer and the actual deberta-v2-xlarge-mnli model
        this model is trained to uunderstand relationships, not generate text
        MNLI = Multi-genre Natural Language Inference
        this model can output 3 Logits: 0, 1 or 2
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(DEVICE)
        self.model.eval()

    
    @torch.no_grad() # so that the model does inference only, no training
    def check_implication(self, text1, text2, example=None):
        '''
        Checks wether text1 entails text2 
        (example is unused here)
        '''
        # tokenization step
        inputs = self.tokenizer(
            text1,
            text2,
            return_tensors="pt",
            truncation=True
        ).to(DEVICE)

        outputs = self.model(**inputs) # calls the deberta mnli model and runs a forward pass based on the inputs
        logits = outputs.logits # extract logits
        probs = F.softmax(logits, dim=1) # convrert logits to probabilities
        prediction = torch.argmax(probs, dim=1).item() # select the most likely class (0, 1 or 2)
        return prediction


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    def are_equivalent(text1, text2):
        '''
        Two strings text1 and text2 are semantically equivalent (with strict_entailment=False) when 
        their implications are either [1, 2], [2, 1] or [2, 2]. With strict_entailment=True, then only [2, 2] means they are equivalent
        '''

        i1 = model.check_implication(text1, text2, example=example)
        i2 = model.check_implication(text2, text1, example=example)

        if strict_entailment:
            return (i1 == 2) and (i2 == 2)
        else:
            implications = [i1, i2]
            return (0 not in implications) and (implications != [1, 1])

    semantic_ids = [-1] * len(strings_list)
    next_id = 0

    # give each string in strings_list a semnatic_id
    for i, s1 in enumerate(strings_list):
        if semantic_ids[i] == -1:
            semantic_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(s1, strings_list[j]):
                    semantic_ids[j] = next_id # give the string the same semantic id if it is equivalent with an other string in strings_list
            next_id += 1

    return semantic_ids
    

'''
LLM CHECKS ENTAILMENT
'''
def LLM_get_semantic_ids(strings_list, sub_question, MODEL_NAME, strict_entailment=False):
    def LLM_check_implication(text1, text2):
        '''
        LLM checks if text1 implies text2. Returns 2 if entailment, 1 if neutral and 0 if contradiction.
        '''
        try:
            client = genai.Client()
        except Exception as e:
            print(f"Error when initializing client: {e}")
            if not os.environ.get("GEMINI_API_KEY"):
                print("ERROR: GEMINI_API_KEY not found in environment.")
            return

        # the prompt below was copied from semantic_uncertainty/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py
        prompt = f"""We are evaluating answers to the question \"{sub_question}\"\n"""
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral, nothing else. """

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,  
                contents=prompt, 
                config=GenerateContentConfig(
                    temperature = 0.7 # temperature = 0 -> models behave deterministically. temperature > 0 -> model randomly choses from top tokens
                )
            )
            entailment_answer = response.lower()[:30]
            if 'entailment' in entailment_answer:
                return 2 
            elif 'neutral' in entailment_answer:
                return 1
            elif 'contradiction' in entailment_answer:
                return 0
            else:
                raise ValueError(
                    f"Unexpected model output: {entailment_answer}. Answer can only be 'entailment, 'neutral' or 'contradiction'. "
                )
        
        except Exception as e:
            raise RuntimeError(
                f"API call failed for prompt: {prompt}"
            ) from e

    def LLM_are_equivalent(text1, text2):
        '''
        Returns True if text1 and text2 are semantically identical. 
        Two strings text1 and text2 are semantically equivalent (with strict_entailment=False) when 
        their implications are either [1, 2], [2, 1] or [2, 2]. With strict_entailment=True, then only [2, 2] means they are equivalent
        '''
        try: 
            i1 = LLM_check_implication(text1, text2)
        except ValueError as e:
            print(f"LLM failed in checking implication of \n'{text1}' --> '{text2}'. \nThe error that was raised is: \n{e}")

        try:
            i2 = LLM_check_implication(text2, text1)
        except ValueError as e:
            print(f"LLM failed in checking implication of \n'{text2}' --> '{text1}'. \nThe error that was raised is: \n{e}")

        if strict_entailment:
            return (i1 == 2) and (i2 == 2)
        else:
            implications = [i1, i2]
            return (0 not in implications) and (implications != [1, 1])

    semantic_ids = [-1] * len(strings_list)
    next_id = 0

    # give each string in strings_list a semnatic_id
    for i, s1 in enumerate(strings_list):
        if semantic_ids[i] == -1:
            semantic_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if LLM_are_equivalent(s1, strings_list[j]):
                    semantic_ids[j] = next_id # give the string the same semantic id if it is equivalent with an other string in strings_list
            next_id += 1

    return semantic_ids

    


