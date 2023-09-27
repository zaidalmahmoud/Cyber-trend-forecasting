# PTs Extraction
This is a Python implementation for the two algorithms called Extractive-GPT (**E-GPT**) and Direct-GPT (**D-GPT**). Both algorithms return the list of pertinent technologies (PTs), which are relevant to a given cyber-attack through interacting with the GPT-3 model. The algorithm E-GPT does so by first collecting research abstracts from Elsevier's database and then prompting GPT to extract the PTs from these abstracts, followed by ranking the answers. The algorithm D-GPT asks GPT a direct question (e.g., What are the PTs relevant to the given attack?). Both algorithms filter out unwanted terms. These terms are stored in the file *excluded_keywords.txt*. Within the prompt, the GPT model is given an example of a question and its corresponding answer to enhance its performance.

## E-GPT
The implementation for the algorithm E-GPT is provided in the file **E-GPT.py**. The output is saved to the file **E-GPT.txt**.

## D-GPT
The implementation for the algorithm D-GPT is provided in the file **D-GPT.py**. The output is saved to the file **D-GPT.txt**.

## Notes/Requirements
For running **E-GPT.py** and **D-GPT.py**, the OpenAI key should be placed in the first line of code. Also, for Elsevier API to be accessed, you should obtain Elsevier key and place it in the file **config.json**.
