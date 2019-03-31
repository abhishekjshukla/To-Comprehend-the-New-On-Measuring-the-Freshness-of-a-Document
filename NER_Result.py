
# coding: utf-8


from nltk import ngrams
import os
import numpy as np
import pickle
import time
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordNERTagger
from gensim.models.doc2vec import Doc2Vec
from gensim.summarization import summarize
from gensim.summarization import keywords
from gensim.models import KeyedVectors
from pyemd import emd
import string
from scipy.spatial.distance import cosine
import math
import pandas as pd





dlnd_path = "../TAPNew/"
all_direc = [dlnd_path+direc+"/"+direc1 for direc in os.listdir(dlnd_path) if os.path.isdir(dlnd_path+direc) for direc1 in os.listdir(dlnd_path+direc)]


for i in xrange(len(all_direc)):
    all_direc[i]+=("/")

source_files = [[direc+"source/"+file for file in os.listdir(direc+"source/") if file.endswith(".txt") and not file.startswith('.')] for direc in all_direc]

target_files = [[direc+"target/"+file for file in os.listdir(direc+"target/") if file.endswith(".txt") and not file.startswith('.')] for direc in all_direc]





ner = StanfordNERTagger("./stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz","./stanford-ner-2017-06-09/stanford-ner.jar")





from collections import defaultdict






source_All=[]
for i in range(len(source_files)):
    for j in range(len(source_files[i])):
        target=open(source_files[i][j],"r").read().decode("ascii","ignore")
        source_All.append(target)





sourcr_tag=[]
j=0
for mn in source_All:
    source_tokens = [token for token in word_tokenize("".join(mn))]
    source_tagged = [i for i in ner.tag(source_tokens) if i[1]!='O']
    sourcr_tag.append(source_tagged)
    j+=1




# save the ner tags
with open('nertag_source_docs.pickle', 'wb') as handle:
    pickle.dump(sourcr_tag,handle, protocol=pickle.HIGHEST_PROTOCOL)


def ner_score(target_text,xx,yy):
    global ner,ner_score_lst,final_ner
    global source_All
    target_tokens = [token for token in word_tokenize(target_text)]
    target_tagged = [i for i in ner.tag(target_tokens) if i[1]!='O']
    for source_tagged in sourcr_tag:
        if len(target_tagged)==0 or len(source_tagged)==0:
            return 0
        else:
            ner_score_val = len([i for i in target_tagged if i in source_tagged])/float(len(target_tagged))
        ner_score_lst[xx,yy].append(ner_score_val)





ner_score_lst=defaultdict(list)
final_ner=defaultdict(list)
ner_score_all=[]
for i in range(len(target_files)):
    for j in range(len(target_files[i])):
        target=open(target_files[i][j],"r").read().decode("ascii","ignore")
        ner_score(target,i,j)



fd=open("result_ner_path_final.csv","w")
for n in ner_score_lst.keys():
    i=n[0]
    j=n[1]
    if(ner_score_lst[i,j]!=[]):
        fd.write((target_files[i][j])+",")
        for k in range(741):
            try:
                fd.write(str(ner_score_lst[i,j][k])+",")
            except:
                print("index",i,j)
        fd.write("\n")
fd.close()



fd2=open("result_ner_path_final.csv","r+")
content = fd2.read()
fd2.seek(0, 0)

fd3=open("src_file_path.csv","r+")
rd=fd3.read()
wr=rd.split(',')


for i in wr:
    fd2.write(i+",")
fd2.write("NNN")
fd2.write("\n")
fd2.close()


df=pd.read_excel("result_ner_path_final.xlsx")


#writing top 10 Results
fd2=open("result_from_ner_withpath.csv","w")
for i in range(len(aa)):
    b=aa[i].sort_values(ascending=[False])
    ind=b.index[0:11].values
    fd2.write(aa[i][0]+",")
    for j in range(10):
        fd2.write(ind[j+1]+",")
    fd2.write("\n")
    

