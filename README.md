# EHRI-NER

A named entity recognition dataset built from EHRI digital editions (https://www.ehri-project.eu/ehri-online-editions).

Documents are sorted by languages.

* All sub-directory: all annotated documents merged.
* Docs sub-directory: each annotated document.

## Annotation format

Each word has been put on a separate line and there is an empty line after each sentence. 

The annotations follow the conll2003 format (IOB).

## Annotation schema

* PER
* LOC
* ORG
* TER (EHRI terms)