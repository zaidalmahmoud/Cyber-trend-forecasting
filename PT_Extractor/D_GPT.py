import os
import openai
import wandb
import random
from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch
import json
import sys
import csv
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd


#configs

openai.api_key = "" #place your key here.

def prompt_gpt3_for_attack_solutions(attack):

    note="Provide 10 answers. Please do not write any additional request. Only answer the question I asked."
    example="Q. Give me the list of research solutions keywords for defending against SQL Injection attack."+note+"\n\n A: Input Validation, Parameterized Queries, Stored Procedures, Escaping, Penetration Testing, Next-Generation Firewall, Web Application Firewall, Object-Relational Mapping, Database Auditing, Content Security Policy"
    #example+="Q. Give me the list of research solutions keywords for defending against Adversarial Machine Learning attack."+note+"\n\n A: Adversarial Training, Data Augmentation, Gradient Masking, Input Regularisation, Ensemble Adversarial Learning, Feature Squeezing, Noise Injection, Defensive Distillation, Large Language Models, Trustworthy AI"
    gpt_prompt =example+ "Q. Give me the list of research solutions keywords for defending against "+attack+" attack."+note+"\n\n A:"
    
    gpt_prompt = gpt_prompt.replace('attack attack','attack')
    gpt_prompt = gpt_prompt.replace('Attack attack', 'attack')


    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=gpt_prompt,
    temperature=1,
    max_tokens=2048,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    print('RESPONSE:')
    print(response['choices'][0]['text'])

    answer=response['choices'][0]['text'];

    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt='Extract the list of solutions keywords from the following text and separate the items by comma and space:\n\n'+answer,
    temperature=0,
    max_tokens=2048,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    answer=response['choices'][0]['text']

    #1. remove numbering 
    if(len(answer)>1 and str(answer[0]).isdigit() and answer[1]=='.'):
        for i in range(30):
            answer=answer.replace(str(i+1)+'. ','')
    #2. replace dash in the middle of terms by space
    for i in range(len(answer)):
        if answer[i]=='-' and i>0 and str(answer[i-1]).isalpha():
            answer=answer[:i]+' '+answer[i+1:]

        
    #3. replace dash by empty string otherwise since it would be at the beginning, and do other replacements then split         
    answerList=answer.replace('-','').replace('\n',', ').split(', ')
    answerList=[s.strip() for s in answerList]


    return [answer for answer in answerList if answer != '']


def is_exclude_solution(solution, attack):
    with open('excluded_keywords.txt') as file:
        lines = [line.rstrip() for line in file]

    solution=solution.upper().replace('-',' ')
    exclude=False
    for line in lines:
        if line.upper().replace('-',' ')== solution:
            exclude=True
            break
        if line.upper().replace('-',' ')+'S'== solution:
            exclude=True
            break
        if line+' attack'.upper().replace('-',' ')== solution:
            exclude=True
            break
        if line+' attack'.upper().replace('-',' ')+'S'== solution:
            exclude=True
            break
        
    if solution==attack.upper().replace('-',' ')+' DETECTION' or solution==attack.upper().replace('-',' ')+' ATTACK DETECTION':
        exclude=True
    if solution==attack.upper().replace('-',' ')+' PREVENTION' or solution==attack.upper().replace('-',' ')+' ATTACK PREVENTION':
        exclude=True
    if solution==attack.upper().replace('-',' ')+' PROTECTION' or solution==attack.upper().replace('-',' ')+' ATTACK PROTECTION':
        exclude=True
    if solution==attack.upper().replace('-',' ')+' MITIGATION' or solution==attack.upper().replace('-',' ')+' ATTACK MITIGATION':
        exclude=True
    if solution==attack.upper().replace('-',' ')+' DEFENSE' or solution==attack.upper().replace('-',' ')+' ATTACK DEFENSE':
        exclude=True
    if solution==attack.upper().replace('-',' ')+' DEFENCE' or solution==attack.upper().replace('-',' ')+' ATTACK DEFENCE':
        exclude=True
    if solution.endswith('ATTACK') or solution.endswith('ATTACKS') or solution.endswith('SECURITY') or solution.endswith(attack) or solution.endswith(attack+'S') or solution.endswith('HIJACKING'):
        exclude=True

    return exclude


attacks=['Supply Chain','Backdoor','Account Hijacking','Botnet','Session Hijacking','Data Poisoning',
         'Trojan','IoT Attack','Deepfake','Ransomware','Malware','Brute Force Attack',
         'Cryptojacking','Adversarial Attack','MITM','Password Attack','DDoS','DNS Spoofing','Advanced Persistent Threat',
         'Dropper','Disinformation','Insider Threat','Phishing','Targeted Attack','Vulnerability','Zero-day']



done=[]



for attack in attacks:
    print('ASKING DIRECT Q FOR:',attack)
    all_solutions=prompt_gpt3_for_attack_solutions(attack)
    solutions=[]
    for s in all_solutions:
        if len(solutions)<10 and not is_exclude_solution(s,attack):
            solutions.append(s)

    print('PTs:\n',solutions,'\n')

    with open('D_GPT.txt', 'a') as the_file:
        the_file.write(attack.upper()+', ')
        for i,v in enumerate(solutions):
            the_file.write(v.upper())
            if i==len(solutions)-1:
                the_file.write('\n')
            else:
                the_file.write(', ')
    time.sleep(40)
with open('D_GPT.txt', 'a') as the_file:
    the_file.write('\n')
