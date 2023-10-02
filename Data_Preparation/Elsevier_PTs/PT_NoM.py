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
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import pandas as pd
from webdriver_manager.firefox import GeckoDriverManager



   
## Load configuration
con_file = open("config.json")#Place your Elsevier key in this file
config = json.load(con_file)
con_file.close()

## Initialise client
client = ElsClient(config['apikey'])

#harvester config (used to count the number of mentions)
options = Options()
options.add_argument('--headless')
driver = webdriver.Firefox(options=options, service=Service(GeckoDriverManager().install()))#executable_path = 'C:/Program Files/Mozilla Firefox/geckodriver')



# this list will be used in the query of Elsevier (PT list)
solution_list=['BLOCKCHAIN',
'"ACCESS CONTROL"',
'ENCRYPTION',
'"SUPPLY CHAIN" AND "RISK MANAGEMENT"',
'"IDENTITY MANAGEMENT"',
'"Double Patterning Lithography"',
'"MACHINE LEARNING"',
'"ANOMALY DETECTION"', 
'CRYPTOGRAPHY',
'"PENETRATION TESTING"',
'"intrusion detection" OR "intrusion prevention"',
'"STATIC ANALYSIS"',
'"DYNAMIC ANALYSIS"',
'"MULTI FACTOR AUTHENTICATION" OR "MULTIFACTOR AUTHENTICATION" OR "MULTI-FACTOR AUTHENTICATION"', 
'"LEAST PRIVILEGE"',
'"SESSION MANAGEMENT"',
'CAPTCHA',
'BLACKLISTING',
'"RATE LIMITING"',
'"GRAPHICAL MODEL*"',
'HONEYPOT',
'"SOFTWARE DEFINED NETWORK"',
'"GAME THEORY"',
'"GRAPH MACHINE LEARNING" OR "GRAPH-BASED MACHINE LEARNING"',
'"IP WHITELIST*"',
'"TRAFFIC SHAPING"',
'"PACKET FILTERING"',
'BLACKHOLING OR "BLACK HOLING" OR ("Black hol*" AND network)',
'"RANK CORRELATION"',
'D3NS AND DNS',
'"SESSION ID RANDOMIZATION" OR "SESSION ID RANDOMISATION"',
'"STRONG AUTHENTICATION" OR Kerberos',
'"SECURE SOCKETS LAYER" OR "Transport Layer Security"',
'{HTTPS}',
'"CONTINUOUS AUTHENTICATION"',
'"Identity-Based Encryption" OR "Identity Based Encryption"',
'"DATA SANITIZATION" OR "DATA SANITISATION"',
'"OUTLIER DETECTION"',
'"DATA PROVENANCE"',
'"ADVERSARIAL TRAINING" OR "adversarial learning"',
'"TRUSTWORTHY AI"',
'"DEEP PROBABILISTIC MODEL*"',
'"Bayesian Network*"',
'"TROJAN ISOLATION"',
'"HARDWARE SANDBOXING"',
'"BEHAVIOR BASED DETECTION" OR "BEHAVIOR-BASED DETECTION" OR "BEHAVIOUR BASED DETECTION" OR "BEHAVIOUR-BASED DETECTION"',
'"FORMAL VERIFICATION"',
'"SPLIT MANUFACTURING"',
'"VULNERABILITY MANAGEMENT"',
'"FILE INTEGRITY" AND MONITOR*',
'{VPN}',
'"CSS MATCHING"',
'"URI MATCHING"',
'"PRIVACY PRESERVING"',
'"SECURE BOOT"',
'"MERKLE SIGNATURE"',
'"LIVENESS DETECTION"',
'"AUDIO DEEPFAKE DETECTION"',
'"3D FACE RECONSTRUCTION"',
'BIOMETRICS',
'"DIGITAL WATERMARK"',
'"APPLICATION WHITELISTING"',
'"DATA BACKUP"',
'"HIDDEN MARKOV MODEL"',
'"PATCH MANAGEMENT"',
'"DATA AUGMENTATION"',
'"DIMENSIONALITY REDUCTION"',
'"DEFENSIVE DISTILLATION"',
'"GRADIENT MASKING"',
'{RRAM}',
'"SPATIAL SMOOTHING"',
'"NOISE INJECTION"',
'"TAINT ANALYSIS"',
'"NETWORK SEGMENTATION"',
'"USER BEHAVIOR ANALYTICS" OR "USER BEHAVIOUR ANALYTICS" OR ("USER BEHAVIOR" AND ANALYTICS) OR ("USER BEHAVIOUR" AND ANALYTICS)',
'"DECEPTION TECHNOLOGY"',
'"RISK ASSESSMENT"',
'"LOG CORRELATION"',
'"DYNAMIC RESOURCE MANAGEMENT"',
'"SANDBOXING"',
'"DARKNET MONITORING" OR "MONITOR* DARKNET" OR "DARKNET MONITOR*"',
'"VIRTUAL KEYBOARD*"',
'"CODE SIGN*"',
'"FILE SIGNATURE"',
'"PUBLIC KEY INFRASTRUCTURE"',
'"VOICEPRINT AUTHENTICATION"',
'"MUTUAL AUTHENTICATION"',
'"PASSWORD SALT*"',
'"ONE TIME PASSWORD"',
'"DYNAMIC BINARY INSTRUMENTATION"',
'"ORTHOGONAL OBFUSCATION"',
'"PASSWORD HASH*"',
'{DNSSEC}',
'"CERTIFICATE PINNING"',
'"SECURE SIMPLE PAIRING"',
'"VULNERABILITY ASSESSMENT"',
'{SIEM}',
'"STANDARDIZED COMMUNICATION" OR "STANDARDISED COMMUNICATION"',
'"Control Flow Integrity"',
'"VULNERABILITY SCAN*"',
'"PASSWORD STRENGTH METER*"',
'"PASSWORD MANAGEMENT"',
'"PASSWORD POLICY" OR "PASSWORD POLICI*"',
'"GRAPHICAL AUTHENTICATION"',
'"DATA LOSS PREVENTION"',
'"DATA LEAKAGE DETECTION" OR "DATA LEAKAGE PREVENTION"',
'"ACTIVITY MONITORING"',
'"MOVING TARGET DEFEN*"',
'"KEYSTROKE DYNAMICS"',
'"ATTACK TREE"',
'"Automatic Violation Prevention"',
'"DISTRIBUTED LEDGER*"',
'"SOURCE IDENTIFICATION"',
'"IMAGE RECOGNITION"',
'{MAUVE}',
'"HYPERGAME"',
'"PREBUNKING"',
'"natural language processing" OR "language model*"']


# this list will be used to count the number of mentions of each PT by the harvester (PT keywords list)
mention_list=[('BLOCKCHAIN',),
              ('ACCESS CONTROL',),
              ('ENCRYPTION',),
              ('RISK MANAGEMENT',),
              ('IDENTITY MANAGEMENT',),
              ('Double Patterning Lithography',),
              ('MACHINE LEARNING',),
              ('ANOMALY DETECTION',),
              ('CRYPTOGRAPHY',),
              ('PENETRATION TESTING',),
              ('intrusion detection','intrusion prevention'),
              ('STATIC ANALYSIS',),
              ('DYNAMIC ANALYSIS',),
              ('MULTI FACTOR AUTHENTICATION','MULTIFACTOR AUTHENTICATION','MULTI-FACTOR AUTHENTICATION'),
              ('LEAST PRIVILEGE',),
              ('SESSION MANAGEMENT',),
              ('CAPTCHA',),
              ('BLACKLISTING',),
              ('RATE LIMITING',),
              ('GRAPHICAL MODEL',),
              ('HONEYPOT',),
              ('SOFTWARE DEFINED NETWORK',),
              ('GAME THEORY',),
              ('GRAPH MACHINE LEARNING','GRAPH-BASED MACHINE LEARNING','GRAPH BASED MACHINE LEARNING'),
              ('WHITELIST',),
              ('TRAFFIC SHAP',),
              ('PACKET FILTER',),
              ('BLACKHOL','BLACK HOL'),
              ('RANK CORRELATION',),
              ('D3NS',),
              ('ID RANDOMIZATION','ID RANDOMISATION'),
              ('STRONG AUTHENTICATION',),
              ('SECURE SOCKETS LAYER','Transport Layer Security','SSL','TLS'),
              ('HTTPS',),             
              ('CONTINUOUS AUTHENTICATION',),
              ('Identity-Based Encryption','Identity Based Encryption'), 
              ('DATA SANITIZATION','DATA SANITISATION'),
              ('OUTLIER DETECTION',),
              ('DATA PROVENANCE',),
              ('ADVERSARIAL TRAINING','adversarial learning'),
              ('TRUSTWORTHY AI',),
              ('DEEP PROBABILISTIC MODEL',),
              ('Bayesian Network',),
              ('TROJAN ISOLATION',),
              ('HARDWARE SANDBOXING',),
              ('BEHAVIOR BASED DETECTION','BEHAVIOR-BASED DETECTION','BEHAVIOUR BASED DETECTION','BEHAVIOUR-BASED DETECTION'),
              ('FORMAL VERIFICATION',),
              ('SPLIT MANUFACTURING',),
              ('VULNERABILITY MANAGEMENT',),
              ('FILE INTEGRITY',),
              ('VPN',),
              ('CSS MATCH',),
              ('URI MATCH',),
              ('PRIVACY PRESERV',),
              ('SECURE BOOT',),              
              ('MERKLE SIGNATURE',),
              ('LIVENESS DETECTION',),              
              ('AUDIO DEEPFAKE DETECTION',),
              ('FACE RECONSTRUCTION',),
              ('BIOMETRIC',), 
              ('DIGITAL WATERMARK',),
              ('APPLICATION WHITELISTING',),
              ('Data BACKUP',),
              ('HIDDEN MARKOV',),
              ('PATCH MANAGEMENT',),
              ('DATA AUGMENTATION',),
              ('DIMENSIONALITY REDUCTION',),
              ('DEFENSIVE DISTILLATION',),
              ('GRADIENT MASK',),
              ('RRAM',),
              ('SPATIAL SMOOTHING',),
              ('NOISE INJECTION',),
              ('TAINT ANALYSIS',),
              ('NETWORK SEGMENTATION',),
              ('USER BEHAVIOR ANALYTICS','USER BEHAVIOUR ANALYTICS'),
              ('DECEPTION TECHNOLOGY',),
              ('RISK ASSESSMENT',),
              ('LOG CORRELATION',),
              ('DYNAMIC RESOURCE MANAGEMENT',),
              ('SANDBOXING',),
              ('DARKNET MONITORING',),
              ('VIRTUAL KEYBOARD',),
              ('CODE SIGN',),
              ('FILE SIGNATURE',),
              ('PUBLIC KEY INFRASTRUCTURE',),
              ('VOICEPRINT AUTHENTICATION',),
              ('MUTUAL AUTHENTICATION',),
              ('PASSWORD SALT',),
              ('ONE TIME PASSWORD',),
              ('DYNAMIC BINARY INSTRUMENTATION',),
              ('ORTHOGONAL OBFUSCATION',),
              ('PASSWORD HASH',),
              ('DNSSEC',),
              ('CERTIFICATE PINNING',),
              ('SECURE SIMPLE PAIRING',),
              ('VULNERABILITY ASSESSMENT',),
              ('SIEM',),
              ('STANDARDIZED COMMUNICATION','STANDARDISED COMMUNICATION'),
              ('Control Flow Integrity',),
              ('VULNERABILITY SCAN',),
              ('PASSWORD STRENGTH METER',),
              ('PASSWORD MANAGEMENT',),
              ('PASSWORD POLICY','PASSWORD POLICIES'),
              ('GRAPHICAL AUTHENTICATION',),
              ('DATA LOSS PREVENTION',),
              ('DATA LEAKAGE DETECTION','DATA LEAKAGE PREVENTION'),
              ('ACTIVITY MONITORING',),
              ('MOVING TARGET DEFEN',),
              ('KEYSTROKE DYNAMICS',),
              ('ATTACK TREE',),
              ('Automatic Violation Prevention',),
              ('DISTRIBUTED LEDGER',),
              ('SOURCE IDENTIFICATION',),
              ('IMAGE RECOGNITION',),
              ('MAUVE',),
              ('HYPERGAME',),
              ('PREBUNKING',),
              ('NATURAL LANGUAGE PROCESS','language model','NLP','LLM')]             
              


years_list=list(range(2011,2023));
month_list=['January','February','March','April','May','June','July','August','September','October','November','December']

empty='Result set was empty'

search_results_list=[]

#Part 1 - collect the URLs of all relevant documents (for each month and each PT)
#The search is within the title, abstract, and keywords
for year in years_list:
    print ("searching in year:", str(year))
    
    for month in month_list:

        #Our dataset starts from July 2011
        if year==2011 and month in month_list[:6]:
            continue;

        
        print ("searching in year/month:", str(year),"/",month)
        slsm=[]
        for index,solution in enumerate(solution_list):
            slm=[]
            print("searching for ",solution)
            doc_srch = ElsSearch("(TITLE("+solution+") OR ABS("+solution+") OR KEY("+solution+")) AND (TITLE(*secur*) OR ABS(*secur*) OR KEY(*secur*)) AND (PUBDATETXT("+month+" "+str(year)+"))",'scopus')
            doc_srch.execute(client, get_all = True)
            if(empty in str(doc_srch.results)):
                #print(doc_srch.results)
                slsm.append(slm);
                continue
            else:
                #print("got" , len(doc_srch.results))
                for col in doc_srch.results:
                    slm.append(col)
                slsm.append(slm);
        search_results_list.append(slsm)
        
  




#Part 2 - Harvesting and counting the number of mentions of each PT during each month
#The counting is within the title, abstract, and keywords
print ("--------------------------Harvesting Time ---------------------------")
data=[]
for index,row in enumerate(search_results_list):
    print("Out of", len(search_results_list)," Harvesting for column ",index)
    attack_map = {}
    for mention in mention_list:
        attack_map[mention[0]]=0;
    for j, col in enumerate(row):
        for k, result in enumerate(col):
            url=result['link'][2]['@href']
            print("URL:",url)
            try:
                driver.get(url)
            except:
                time.sleep(20)
                driver.get(url)
            time.sleep(1)
            delay = 20 # seconds
            try:
                myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'Highlight-module__akO5D')))
            except TimeoutException:
                print ("Loading took too much time!")
            #click index keywords
            keyword_elements =  driver.find_elements(By.TAG_NAME,"button")
            keyword_element  = None
            while(True):
                failed=False
                for element in keyword_elements :
                    try:
                        if('Indexed keywords' in element.text):
                            keyword_element=element
                            print("keyword element found!")
                            break;
                    except:
                        print("button issue")
                        keyword_element=None
                        failed=True
                        time.sleep(1)
                        break;
                if failed==False:
                     break
                else:
                     keyword_elements =  driver.find_elements(By.TAG_NAME,"button")

            if(keyword_element is not None) :       
                keyword_element.click()
                time.sleep(1)
            mention = mention_list[j]
            
            #counting
            while (True):
                failed=False
                totalQ=0;
                text_elements =  driver.find_elements(By.CLASS_NAME,"Highlight-module__akO5D");
                for element in text_elements:
                    try:
                        element_text=element.text
                    except:
                        failed=True;
                        break;
                    count = sum(element_text.replace("-"," ").upper().count(x.upper().replace("-"," ")) for x in mention)
                    print("count:",count)
                    totalQ+=count
                if(not failed):
                    break;
            attack_map[mention[0]]+=totalQ
            print("for solution",mention,"sum is now:",  attack_map[mention[0]])
    data_list=[]
    for key in attack_map:
        print(key,": ",attack_map[key])
        data_list.append(attack_map[key])
    data.append(data_list)  
    
    #save data to file
    file_object = open('PTs_NoM.txt', 'a')
    for ii,item in enumerate(data_list):
        file_object.write(str(item))
        if ii<len(data_list)-1:
            file_object.write("\t")
    file_object.write("\n")
    file_object.close()