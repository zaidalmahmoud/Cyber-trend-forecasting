# Data Preparation Overview
This directory contains 5 Python scripts for building the features of our dataset. The features include monthly records for the following:

1- The number of cyber-attack incidents in each country for each attack type (NoI)<br>
2- The number of mentions of each attack type in Elsevier abstracts (A_NoM)<br>
3- The number of mentions of each Pertinent Technology (PT) in Elsevier abstracts (PT_NoM)<br>
4- The number of tweets about wars and political conflicts related to each country (ACA)<br>
5- The number of public holidays in each country (PH)<br>

## Hackmageddon_Attacks
This directory contains the scripts **NoI_daily.py** and **NoI_monthly.py**, which extract the feature **NoI**. The script in the file **NoI_daily.py** reads [Hackmageddon](https://www.hackmageddon.com/) dataset in the file **Hackmageddon.csv**, and transforms textual Hackmageddon data to numerical format by counting the number of incidents (NoI) for each attack type and for each country. The output is the daily NoI between July 2011 and December 2022. The output is saved to the file **NoI_daily.csv**. The script in the file **NoI_monthly.py** reads the daily NoI in the file **NoI_daily.csv** and transforms it to the monthly NoI. The output is saved to the file **NoI_monthly.csv**.  

## Elsevier_Attacks
This directory contains the script **A_NoM.py**, which extracts the feature **A_NoM** using Elsevier API. The input to the script is the list of attack types hardcoded at the beginning of the script (please see the script for the detailed format). The output is the monthly mentions of each attack type in Elsevier research abstracts between July 2011 and December 2022. The output is saved to the file **Attacks_NoM.txt**, where each column in the file corresponds to a single attack type, according to the order provided, and each row corresponds to the month starting from July 2011.

## Elsevier_PTs
This directory contains the script **PT_NoM.py**, which extracts the feature **PT_NoM** using Elsevier API. The input to the script is the list of PTs hardcoded at the beginning of the script (please see the script for the detailed format). The output is the monthly mentions of each PT in Elsevier research abstracts between July 2011 and December 2022. The output is saved to the file **PTs_NoM.txt**, where each column in the file corresponds to a single PT, according to the order provided, and each row corresponds to the month starting from July 2011.

## Python_Holidays
This directory contains the script **PH.py**, which extracts the feature **PH** using the Python Holidays package. This script produces the csv file **PH.csv** with the monthly number of public holidays in 36 countries in the period between July 2011 until December 2022.


## Notes/Requirements
For running **A_NoM.py** and **PT_NoM.py**, you should obtain Elsevier key and place it in the file **config.json** within the script directory.

