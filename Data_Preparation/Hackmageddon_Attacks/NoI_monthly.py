"""
Created on Jan 20 01:18:45 2023

@author: Zaid Almahmoud

The script below transforms the daily number of incidents (NoI) to monthly number of incidents

Input: NoI_daily.csv
Output: NoI_monthly.csv

"""

import sys
import csv

col=988
months=['01','02','03','04','05','06','07','08','09','10','11','12']

def is_leap_year(year):
    """Determine whether a year is a leap year."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


input_file='NoI_daily.csv'
output_file='NoI_monthly.csv'

datafile = open(input_file, 'r')
data = list(csv.reader(datafile))


header=data[0]
data=data[1:]


for row in data:
    del row[0] 


data_m=[]
index=-1
for year in range(2011,2023):
    for month in months:

        #not included in our data   
        if year==2011 and int(month)<7:
            continue;
             
        counter=0
        m=[month+'/'+str(year)]
        m+=[0]*col
        for day in range(1,32):
            
         if month=='02' and day>28 and not is_leap_year(year):
             continue;
         if month=='02' and day>29:
             continue;
         if day>30 and (month=='04' or month=='06' or month=='09' or month=='11'):
            continue;
         
            
         a=''
         if day<10:
             a='0'
         date=a+str(day)+"/"+month+"/"+str(year);
         
         index+=1
         for j in range(0,col):#columns
            value=float(m[j+1])+float(data[index][j])
            m[j+1]=str(value)   
                 
        data_m.append(m) 
        
data_m.insert(0,header[0:col+1])


with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data_m)
         
         
         
         
         
         
         
         
         
         