import os
import re

from bs4 import BeautifulSoup
import spacy
from spacy.attrs import ORTH, NORM
import pandas as pd

# TODO: refactor

DISABLED = [
    "ner", "tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]

# Variables to change depending on corpus
corpus_directory = './tei'
nlp = spacy.load("xx_sent_ud_sm")
nlp.tokenizer.add_special_case('etc.', [{ORTH: 'etc.', NORM: 'etc.'}])
nlp.tokenizer.add_special_case('Mr.', [{ORTH: 'Mr.', NORM: 'Mr.'}])
nlp.tokenizer.add_special_case('Mrs.', [{ORTH: 'Mrs.', NORM: 'Mrs.'}])
nlp.tokenizer.add_special_case('Dr.', [{ORTH: 'Dr.', NORM: 'Dr.'}])

# sentence segmentation
nlp.add_pipe('sentencizer')

def type_to_iob(enttype, idx):
    mapping = {
        "persName": "PERS",
        "placeName": "LOC",
        "orgName": "ORG",
        "date": "DATE",
        "ghetto": "GHETTO",
        "camp": "CAMP"
    }
    iob = 'B' if idx == 0 else 'I'
    return '{}-{}'.format(iob, mapping.get(enttype))

def transform_to_iob(item):
    tokens_spacy = list(nlp(item.text.replace("\n", "").strip(), disable=DISABLED))
    tokens = []
    for token in tokens_spacy:
            tokens.append(token)

    return [
        (ent, type_to_iob(item.name, idx))
        for idx, ent in enumerate(tokens)
    ]

# we create a list for storing [token, label]
annotations = []

# We create a dictionary for storing entities count
stats = {'entity': 
{'PER': 0, 'LOC': 0, 'ORG': 0, 'CAMP': 0, 'DATE': 0, 'GHETTO': 0},
'token': 0}

# initiatilising counters
PER_counter = 0
LOC_counter = 0
ORG_counter = 0
DATE_counter = 0
CAMP_counter = 0
GHETTO_counter = 0

for subdir, dirs, files in os.walk(corpus_directory):
    for file in files:
        print(f'Processing {file} in sub-directory {subdir}')
        with open(os.path.join(subdir, file), mode='r', encoding='utf-8') as fh:
            # parsing the XML-TEI file
            # Getting rid of multiple spaces
            xml_str = fh.read()
            text=re.sub("\s\s+", " ", xml_str)
            # Getting rid of space at the beginning of text
            text = re.sub("^\s", "", text)         
            soup = BeautifulSoup(text, features='xml')
            # looking for <body>
            for elem in soup.find_all("body"):
                # by default, we iterate on all <p>s, can be modified depending on the TEI tree
                for p in elem.find_all("p"):
                    # we iterate on all children of <p>
                    for child in p.children:
                        # same logic is used for remaining entities
                        if child.name == 'placeName':
                            if child.has_attr('type'):
                                if child['type'] == 'camp':
                                    CAMP_counter += 1 
                                    child.name = 'camp'
                                    for iob_tag in transform_to_iob(child):
                                        annotations.append([str(iob_tag[0]), iob_tag[1]])
                                if child['type'] == 'ghetto':
                                    GHETTO_counter += 1 
                                    child.name = 'ghetto'
                                    for iob_tag in transform_to_iob(child):
                                        annotations.append([str(iob_tag[0]), iob_tag[1]])
                            else:
                                LOC_counter += 1 
                                for iob_tag in transform_to_iob(child):
                                    annotations.append([str(iob_tag[0]), iob_tag[1]])

                        elif child.name == 'orgName':
                            ORG_counter += 1
                            for iob_tag in transform_to_iob(child):
                                annotations.append([str(iob_tag[0]), iob_tag[1]])
                        
                        elif child.name == 'date':
                            DATE_counter += 1
                            for iob_tag in transform_to_iob(child):
                                annotations.append([str(iob_tag[0]), iob_tag[1]])

                        elif child.name == 'persName':
                            PER_counter += 1
                            for iob_tag in transform_to_iob(child):
                                annotations.append([str(iob_tag[0]), iob_tag[1]])

                        else:
                            label = 'O'
                            doc = nlp(str(child.text).replace("\n", "").strip(), disable=DISABLED)
                            for sentence in doc.sents:
                                for i, token in enumerate(sentence):
                                    if i == len(sentence)-1 and token.text == '.' or token.text == '?' or token.text == '!' or token.text == ';' or token.text == ':':
                                        annotations.append([str(token), label]) 
                                        annotations.append(['nAn'])
                                    else:
                                        annotations.append([str(token), label])

# updating the entities count
stats['entity']['LOC'] += LOC_counter
stats['entity']['PER'] += PER_counter
stats['entity']['ORG'] += ORG_counter
stats['entity']['DATE'] += DATE_counter
stats['entity']['GHETTO'] += GHETTO_counter
stats['entity']['CAMP'] += CAMP_counter
stats['token'] += len(annotations)

with open(f'./test.txt', mode='w+', encoding='utf-8') as iob_file:
    for annotation in annotations:
        if annotation[0] == 'nAn':
            iob_file.write('\n')
        else:
            iob_file.write(' '.join(annotation)+ '\n')

with open(f'./test_stat.txt', mode='w+', encoding='utf-8') as txt:
    txt.write(str(stats))