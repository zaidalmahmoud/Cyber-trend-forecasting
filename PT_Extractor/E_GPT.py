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

## Load configuration
con_file = open("config.json")
config = json.load(con_file)
con_file.close()
client = ElsClient(config['apikey'])


def  getDefaultFirefoxOptions():
    options = Options()
    options.add_argument('--headless')
    options.add_argument("-profile")
    options.add_argument("--marionette-port")
    options.add_argument("2828")
    options.add_argument("--headless")
    options.add_argument("devtools.selfxss.count", 100)

options = Options()
options.add_argument('--headless')
driver = webdriver.Firefox(options=options)



#computes the number of words in a keyword/phrase
def len_kw(keyword):
     keyword_list=keyword.split(' ')
     return len(keyword_list)


#checks whether a keyword (or phrase) is the first part of a list of strings
def keyword_in_list(keyword,text):

    keyword_list=keyword.split(' ')

    if len(keyword_list)>len(text):
        return False
    for i in range(len(keyword_list)):
        if keyword_list[i]!=text[i]:
            return False

    return True


#computes the smallest distance between a keyword and any term in a list of solution synonyms.
#if no such distance found, returns the length of the given text
def compute_min_distance(key_word,abstract):
    with open('solution_synonyms.txt') as file:
        terms = [line.rstrip() for line in file]
    
    words=abstract.upper().replace('-',' ').replace('\n',' ').replace('.','').replace(',','').replace(';','').replace('?','').split(' ')
    keyword=key_word.upper().replace('-',' ').replace('\n',' ').replace('.','').replace(',','').replace(';','').replace('?','')
    #print('words list=',words)

    min_distance=999999999
    distance=0
    for i in range(len(words)):
        if keyword_in_list(keyword,words[i:]): #keyword here? 
            #iterate backwards until you find the term (e.g.,"solution")
            for j in range(i-1,-1,-1):
                if words[j] in terms:
                    distance=i-j
                    if distance < min_distance:
                        min_distance=distance
                    break

            #iterate forward until you find the term (e.g.,"overcome")
            for j in range(i+len_kw(keyword),len(words)):
                if words[j] in terms:
                    distance=j-((i+len_kw(keyword))-1)
                    if distance < min_distance:
                        min_distance=distance
                    break
    if min_distance==999999999:
        print('DISTANCE NOT FOUND!')
        return len(abstract)

    return min_distance






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
    if solution.endswith('ATTACK') or solution.endswith('ATTACKS') or solution.endswith('SECURITY') or solution.endswith(attack) or solution.endswith(attack+'S') or solution.endswith('THREAT') or solution.endswith('THREATS') or solution.endswith('PROTECTION') or solution.endswith('DETECTION') or solution.endswith('PREVENTION') or solution.endswith('MITIGATION'):
        exclude=True

    return exclude

# This method collects n research abstracts (with title and keywords) from Elsevier related to a given attack.
# The abstract should include mitigation related keywords/pertinent technologies.
# It is recommended that you clear your Firefox browser before running this code.
def query_Elsevier_for_attack_solutions_abstracts(attack,abs_N):
    
    if (not 'attack' in attack.lower()):
        attack+=' attack'


    doc_srch = ElsSearch("(TITLE("+attack+") OR ABS("+attack+") OR KEY("+attack+")) "+
                         "AND PUBYEAR > 2010 "+
                         "AND (TITLE(countermeasure) OR ABS(countermeasure) OR KEY(countermeasure) OR TITLE(mitigat) OR ABS(mitigat) OR KEY(mitigat) OR TITLE(defen) OR ABS(defen) OR KEY(defen) OR ABS(detect) OR KEY(detect) OR TITLE(detect) OR ABS(prevent) OR KEY(prevent) OR TITLE(prevent) OR ABS(protect) OR KEY(protect) OR TITLE(protect) OR ABS(security solution) OR KEY(security solution) OR TITLE(security solution)) ",
                         'scopus')
    
    doc_srch.execute(client, get_all = True)
    empty='Result set was empty'
    slm=None
    if(empty not in str(doc_srch.results)):
        slm=len(doc_srch.results)

    print('length of results=',slm)

    abs_list=[]
    k_index=[-1]*abs_N
    results=None
    if slm != None:
        results=doc_srch.results
        random.shuffle(results) #shuffle the results so we take first 100
        for i in range(0,min(abs_N,len(results))):
            url=results[i]['link'][2]['@href']
            print('document',str(i+1),'URL:',url)
            doc_eid=results[i]['eid']
            print('document',str(i+1),'ID:',doc_eid[doc_eid.index('-',2)+1:])
            
            #harvesting the abstract using the URL
            try:
                driver.get(url)
            except:
                time.sleep(20)
                driver.get(url)
            time.sleep(3)
            delay = 20 # seconds
            try:
                myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'Highlight-module__akO5D')))
                print ("Page is ready!")
            except TimeoutException:
                print ("Loading took too much time!")
            #click index keywords
            keyword_elements =  driver.find_elements(By.TAG_NAME,"button");
            keyword_element  = None
            for element in keyword_elements :
                try:
                    if('Indexed keywords' in element.text):
                        keyword_element=element
                        print("keyword element found!")
                        break;
                except:
                        print("button issue")
                        keyword_element=None
                        continue
            if(keyword_element is not None):       
                keyword_element.click()
                time.sleep(3)
            
            abstract=''
            text_elements =  driver.find_elements(By.CLASS_NAME,"Highlight-module__akO5D");
            for element in text_elements:
                abstract+=element.text+'\n'
                #if not interested in keywords:
                if abstract.count('\n')==2:
                    k_index[i]=len(abstract)
            
            #not paper
            if abstract.count(';')>3:
                continue
                
            abs_list.append(abstract)#finally append the title, abstract, keywords to the list
            print(abstract)
            print('\n','--'*20,'\n')

    return abs_list,k_index
    
    
        
# Given an abstract and an attack type, this method prompts GPT3 to extract technology solutions pertinent to the given attack from the abstract.
def prompt_GPT3_to_extract_technology_solutions(attack, abstract):

    #example 1
    with open('abs0.txt') as f:
        abs0 = f.read()
    #example 2
    with open('abs1.txt') as f:
        abs1 = f.read()

    note='Please only provide keywords from the provided abstract text.'
    example='Q: Can you extract the list of technology solutions keywords to Backdoor attack from the following text\n\n'+abs0+'\n\n'+note+'\n\nSolutions: secured SSL training, Purification, Detection and purification\n\n'
    example+='Q: Can you extract the list of technology solutions keywords to ransomware attack from the following text\n\n'+abs1+'\n\n'+note+'\n\nSolutions: data backup, Data recovery, Data isolation, gateways\n\n'

    gpt_prompt = example+"Q: Can you extract the list of technology solutions keywords to "+attack+" attack from the following text\n\n"+abstract+'\n\n'+note+'\n\nSolutions:'
    


    gpt_prompt = gpt_prompt.replace('attack attack','attack')
    gpt_prompt = gpt_prompt.replace('Attack attack', 'attack')

    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=gpt_prompt,
            temperature=1,
            max_tokens=2048,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    except:
        return []


    answer=response['choices'][0]['text']

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt='Extract the list from the following text, and separate items by comma and space:\n\n'+answer,
        temperature=0,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    answer=response['choices'][0]['text']

    print('\n********************************\n')
    print('GPT answer:')
    print(attack,'Relevant Technology Keywords:',answer+'\n')


    answerList=answer.replace('\n',', ').split(', ')
    answerList=[s.strip() for s in answerList]
    answerList=list(dict.fromkeys(answerList))#unique elements only

    #exclude empty answers
    answerList= [answer for answer in answerList if answer != '' and answer!=',']

    #exclude wrong answers
    answerList= [answer for answer in answerList if answer.lower() in abstract.lower()]

    return answerList




#returns top N relevant technology solutions to a given attack
def iterative_prompt(attack,N,abs_N,relax):
    #k_index is the list of first index of author keywords in the text of each abstract
    abstract_list,k_index= query_Elsevier_for_attack_solutions_abstracts("\""+attack+"\"",abs_N)
    rank= {} #empty_dictionary
    for i,abstract in enumerate(abstract_list):
        keywords=prompt_GPT3_to_extract_technology_solutions(attack, abstract)
        print('keywords of abstract',str(i+1),keywords)
        time.sleep(40)#To avoid GPT rate limit error
        for key_word in keywords:
            if not is_exclude_solution(key_word,attack):
                min_distance_=compute_min_distance(key_word,abstract[:k_index[i]])
                if not key_word.upper() in rank:
                    rank[key_word.upper()]=[min_distance_,1,0]
                else:
                    rank[key_word.upper()]=[rank[key_word.upper()][0]+min_distance_,#smallest distance
                                                rank[key_word.upper()][1]+1, #frequency
                                                0]
    #average and assign final rank                                        
    for key in rank:
        rank[key][0]/=rank[key][1] #average smallest distance
        rank[key][2]=rank[key][0]/rank[key][1]#average smallest distance/frequency (not used)

    sorted_rank ={k: v for k, v in sorted(rank.items(), key=lambda item: (item[1][1],-item[1][0]))}
    print('sorted rank:')
    for key in sorted_rank:
        print(key,": ",sorted_rank[key]) 

    sorted_rank_list=[key for key in sorted_rank]
    if len(sorted_rank_list)<N:
        return sorted_rank_list
    return sorted_rank_list[-N:]
        




attacks=['Supply Chain','Backdoor','Account Hijacking','Botnet','Session Hijacking','Data Poisoning',
         'Trojan','IoT Attack','Deepfake','Ransomware','Malware','Brute Force Attack',
         'Cryptojacking','Adversarial Attack','MITM','Password Attack','DDoS','DNS Spoofing','Advanced Persistent Threat',
         'Dropper','Disinformation','Insider Threat','Phishing','Targeted Attack','Vulnerability','Zero-day']


#can be used if the program crashed for any reason and the PTs of some attacks have been already extracted.
done=[]


N=10
abs_N=50
relax=0


for attack in attacks:
    if attack in done:
        continue


    PTs=iterative_prompt(attack, N,abs_N,relax)
    print('\nPTs:\n',PTs)

    with open('E_GPT.txt', 'a') as the_file:
        the_file.write(attack.upper()+', ')
        for i,v in enumerate(PTs):
            the_file.write(v.upper())
            if i==len(PTs)-1:
                the_file.write('\n')
            else:
                the_file.write(', ')

#the end
with open('E_GPT.txt', 'a') as the_file:
    the_file.write('\n')




