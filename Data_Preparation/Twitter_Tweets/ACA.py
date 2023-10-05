"""
Created on Jan 21 01:20:38 2023

@author: Zaid Almahmoud

This script counts for each country the number of tweets about wars and conflicts related to that country.
The output is the monthly count of these tweets for each country in the period between July 2011 and December 2022.

Output file: ACA.csv
"""

# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
#To add wait time between requests
import time
from csv import writer


#Place your Twitter API's bearer token here!
os.environ['TOKEN'] = ''


def create_url(keyword, start_date, end_date, max_results = 10):
    
    search_url = "https://api.twitter.com/2/tweets/search/all" #Change to the endpoint you want to collect data from

    #change params based on the endpoint you are using
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)



def auth():
    return os.getenv('TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def is_leap_year(year):
    """Determine whether a year is a leap year."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


months=['01','02','03','04','05','06','07','08','09','10','11','12']
countries_s=['(USA OR America)',
           '(UK OR British OR United Kingdom OR Britain)',
           '(CANADA OR CANADIAN)',
           '(AUSTRALIA)',
           '(Ukraine)',
           '(RUSSIA)',
           '(FRANCE OR FRENCH)',
           '(GERMAN)',
           '(Brazil)',
           '(China OR chinese)',
           '(Japan)',
           '(Pakistan)',
           '(North Korea)',
           '(South Korea)',
           '(India)',
           '(Taiwan)',
           '(NetherLands OR Holland OR Dutch)',
           '(SPAIN OR Spanish)',
           '(Sweden OR Swedish)',
           '(Mexic)',
           '(IRAN)',
           '(ISRAEL)',
           '(Saudi)',
           '(Syria)',
           '(Finland OR FINNISH)',
           '(IRELAND OR IRISH)',
           '(AUSTRIA)',
           '(NORWAY OR Norwegian)',
           '(Switzerland OR swiss)',
           '(ITALY OR ITALIAN)',
           '(MALAYSIA)',
           '(EGYPT)',
           '(TURKEY OR TURKISH)',
           '(portugal OR portuguese)',
           '(Palestin OR West Bank OR GAZA)',
           '(UAE OR United Arab Emirates OR emarat)']

countries=['US','GB','CA','AU','UA','RU','FR','DE','BR','CN','JP','PK',
           'KP','KR','IN','TW','NL','ES','SE','MX','IR','IL','SA','SY',
           'FI','IE','AT','NO','CH','IT','MY','EG','TR','PT','PS','AE']



header=['Date']
for country in countries:
    header.append('WAR/CONFLICT '+country)

conflicts=[header]


with open('ACA.csv','a',newline="") as f:
    for year in range(2011,2023):
        for month in months:

            #not included
            if int(month)<7 and year==2011:
                continue
 
            days=31;
            if (month=='04' or month=='06' or month=='09' or month=='11'):
                days=30
            if (month=='02' and is_leap_year(year)):
                days=29
            if (month=='02' and not is_leap_year(year)):
                days=28


                
            conflict=[month+'/'+str(year)]
                
            c_index=-1
            for country in countries: 
               c_index+=1
               #Inputs for the request
               bearer_token = auth()
               headers = create_headers(bearer_token)
               keyword = "("+countries_s[c_index]+" WAR MILITARY) OR ("+countries_s[c_index]+" WAR ARMED FORCE) OR ("+countries_s[c_index]+" CONFLICT POLITIC) OR ("+countries_s[c_index]+" MILITARY ATTACK) OR ("+countries_s[c_index]+" ARMED FORCE ATTACK) lang:en"
               start_time = str(year)+"-"+month+"-"+'01'+"T00:00:00.000Z"
               end_time = str(year)+"-"+month+"-"+str(days)+"T23:59:59.000Z"
               max_results = 400
       
       
       
               count=0;
               flag=True;
               next_token = None
               while(flag):
                   
                    if count >= 200000:
                        count=200000
                        break
                   
                    print('creatin URL....')
                    url = create_url(keyword, start_time,end_time, max_results)
                    print('getting json response...')
                    json_response = connect_to_endpoint(url[0], headers, url[1],next_token)               
                    result_count = json_response['meta']['result_count']
                   
                    if 'next_token' in json_response['meta']:
                       # Save the token to use for next call
                        next_token = json_response['meta']['next_token']
                        print("Next Token: ", next_token)
                        if result_count is not None and result_count > 0 and next_token is not None:
                            count += result_count
                            print("--------NEXT PAGE-----------",country,month+'/'+str(year)+":", count)
                            time.sleep(6)                
                   # If no next token exists
                    else:
                        if result_count is not None and result_count > 0:
                            print("-------------------")
                            count += result_count
                            print("------FINISHED-------------")
                            print("Total number of results for: ",country,month+'/'+str(year)+":", count)
                            time.sleep(5)
                           
                       #Since this is the final request, turn flag to false to move to the next time period.
                        flag = False
                        next_token = None
                    time.sleep(5)
                   
               conflict.append(count)
            conflicts.append(conflict)
            
            writer_object = writer(f)
            writer_object.writerow(conflict) 
            f.flush()
f.close()
        





