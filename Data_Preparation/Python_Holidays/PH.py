"""
Created on Jan 20 01:49:41 2023

This script produces a csv file "PH.csv" with the monthly number of public holidays in 36 countries in the period July 2011 until December 2022.

@author: Zaid Almahmoud
"""

from datetime import date
import holidays
import sys
import csv

def is_leap_year(year):
    """Determine whether a year is a leap year."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


#example
us_holidays = holidays.country_holidays('US')  # this is a dict
date(2015, 1, 1)in us_holidays  # True
date(2015, 1, 2) in us_holidays  # False
us_holidays.get('2014-01-01')  # "New Year's Day"






countries=['US','GB','CA','AU','UA','RU','FR','DE','BR','CN','JP','PK',
           'KP','KR','IN','TW','NL','ES','SE','MX','IR','IL','SA','SY',
           'FI','IE','AT','NO','CH','IT','MY','EG','TR','PT','PS','AE']


H=[]

months=['01','02','03','04','05','06','07','08','09','10','11','12']



h=['date']
for year in range(2011,2023):
    for month in months: 
             
            #not included
            if int(month)<7 and year==2011:
                continue;
            
            date_s=month+"/"+str(year);
            h.append(date_s)

H.append(h)



missing=[]
for country in countries:
    h=[country+'_holiday']
    for year in range(2011,2023):
        for month in months:
            
            #not included
            if int(month)<7 and year==2011:
                continue;
            counter=0


            for day in range(1,32):
            
                if month=='02' and day>28 and not is_leap_year(year):
                    continue;
                if month=='02' and day>29:
                    continue;
                if day>30 and (month=='04' or month=='06' or month=='09' or month=='11'):
                    continue;
            

                try:
                    c_holidays = holidays.country_holidays(country)  # this is a dict
                except:
                    if not country in missing:
                        missing.append(country)
                    continue;

                if(date(year, int(month), day) in c_holidays):
                    counter+=1

            h.append(counter)
    
    H.append(h)
    print('Added holidays of',country)




with open('PH.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(list(map(list, zip(*H))))




