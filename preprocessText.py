import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import nltk

#nltk.download('stopwords')

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

def lemmatize(doc):
    doc = re.sub(patterns, '', doc)
    doc = doc.lower()
    doc = re.sub("^\s+|\n|\r|\s+$", '', doc)
    doc = doc.replace("«", "")
    doc = doc.replace("»", "")
    #doc = re.sub(stopwords_ru,'',doc)
    #doc = re.sub(r'\s+', ' ', doc)
    #tokens = []
    #for token in doc.split():
        #if token and token not in stopwords_ru:
            #token = token.strip()
            #token = morph.normal_forms(token)[0]

            #tokens.append(token)
    #if len(tokens) > 2:
        #return tokens
    #return None

    return doc

name1= "5500PagesParsed.txt"
name2= "opCorporaJustText.txt"
nameRes1="5500Clear.txt"
nameRes2="opcorporaClear.txt"


# Cleaing the text
f = open(name1, 'r', encoding='utf-8')
fl = open(name2, 'r', encoding='utf-8')

lines = f.readlines()

# итерация по строкам
fj = open(nameRes1, 'w', encoding='utf-8')
for line in lines:
    i=1
    #st=''
    #a=lemmatize(line)
    #print(a)
    #if not (a is None):
    #    for t in a:
    #        st = st + str(t) + ' '
    #    st= st + '\n'
    #    fj.write(st)
    if len(line) >=10:
        fj.write(lemmatize(line) + '\n')

lines = fl.readlines()

# итерация по строкам
fj = open(nameRes2, 'w', encoding='utf-8')
for line in lines:

    #st = ''
    #a = lemmatize(line)
    #if not (a is None):
        #for t in a:
            #st = st + str(t) + ' '
        #st = st + '\n'
        #fj.write(st)

    fj.write(lemmatize(line) + '\n')

#processed_article = article_text.lower()
#processed_article = re.sub(patterns, ' ', processed_article )
#processed_article = re.sub(r'\s+', ' ', processed_article)