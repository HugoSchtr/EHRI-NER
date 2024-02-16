import os

from bs4 import BeautifulSoup
import spacy

# TODO: refactor

DISABLED = [
    "ner", "tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]

# Variables to change depending on corpus
corpus_directory = '../corpus/lang/tei/sk'
nlp = spacy.load("xx_sent_ud_sm")

# sentence segmentation
nlp.add_pipe('sentencizer')

def type_to_iob(enttype, idx):
    mapping = {
        "persName": "PER",
        "placeName": "LOC",
        "orgName": "ORG",
        "term": "TER",
    }
    iob = 'B' if idx == 0 else 'I'
    return '{}-{}'.format(iob, mapping.get(enttype))

def transform_to_iob(item):
    tokens_spacy = list(nlp(item.text.replace("\n", "").strip(), disable=DISABLED))
    tokens = []
    for token in tokens_spacy:
        if str(token).isspace():
            pass
        else:
            tokens.append(token)

    return [
        (ent, type_to_iob(item.name, idx))
        for idx, ent in enumerate(tokens)
    ]

# we create a list for storing [token, label]
annotations = []

# We create a dictionary for storing entities count
stats = {'entity': 
{'PER': 0, 'LOC': 0,'ORG': 0,'TER': 0},
'token': 0}

# initiatilising counters
token_counter = 0
PER_counter = 0
LOC_counter = 0
ORG_counter = 0
TER_counter = 0


for subdir, dirs, files in os.walk(corpus_directory):
    for file in files:
        print(f'Processing {file} in sub-directory {subdir}')
        with open(os.path.join(subdir, file), mode='r', encoding='utf-8') as fh:
            # parsing the XML-TEI file
            soup = BeautifulSoup(fh, features='xml')
            # looking for <body>
            for elem in soup.find_all("body"):
                # by default, we iterate on all <p>s, can be modified depending on the TEI tree
                for p in elem.find_all("p"):
                    # we iterate on all children of <p>
                    for child in p.children:
                        token_counter += 1
                        # same logic is used for remaining entities
                        if child.name == 'term':
                            TER_counter += 1
                            for iob_tag in transform_to_iob(child):
                                annotations.append([str(iob_tag[0]), iob_tag[1]])
                        
                        elif child.name == 'placeName':
                            LOC_counter += 1
                            for iob_tag in transform_to_iob(child):
                                annotations.append([str(iob_tag[0]), iob_tag[1]])

                        elif child.name == 'orgName':
                            ORG_counter += 1
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
                                for token in sentence:
                                    if str(token).isspace():
                                        pass
                                    else:
                                        annotations.append([str(token), label]) 

# updating the entities count
stats['entity']['LOC'] += LOC_counter
stats['entity']['PER'] += PER_counter
stats['entity']['ORG'] += ORG_counter
stats['entity']['TER'] += TER_counter
stats['token'] += len(annotations)


with open(f'../corpus/lang/conll/sk/all/ehri-ner_sk_all.txt', mode='w+', encoding='utf-8') as iob_file:
    for annotation in annotations:
        if annotation[0] == '.':
            iob_file.write(' '.join(annotation) + '\n\n')
        elif annotation[0] == '"':
            pass
        else:
            iob_file.write(' '.join(annotation)+ '\n')

with open(f'../corpus/lang/conll/sk/all/ehri-ner_sk_all_stats.txt', mode='w+', encoding='utf-8') as txt:
    txt.write(str(stats))
    