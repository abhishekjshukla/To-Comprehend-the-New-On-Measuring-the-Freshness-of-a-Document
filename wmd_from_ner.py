
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
from gensim.models import Word2Vec





stoplist = set(stopwords.words('english') + list(string.punctuation))

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)






import pandas as pd
from collections import defaultdict

df=pd.read_csv("result_ner_path_final.csv",header=None)


dfn=pd.read_csv("result_from_ner_withpath.csv",header=None)





target_files=df[:,0]





sourceF=np.empty((1,10))
for i in range(len(df)):
    sourceF=np.vstack((sourceF,df[i,1:11]))





sourceF=sourceF[1:,:]





dist=defaultdict(list)
final_dist=defaultdict(list)
def wmd_dis(target_files,i):
    target_doc=open(target_files,"r").read().decode("ascii","ignore")
    target_doc = [w for w in target_doc if w not in stoplist]
    target_doc="".join(target_doc)
    k=0
    for j in sourceF[i]:
        doc=open(j,"r").read().decode("ascii","ignore")
        src=[w for w in doc if w not in stoplist]
        src="".join(src)
        distance = model.wmdistance(target_doc, src)
        print(distance,i,j)
        dist[i,k].append(distance)
        k+=1
    print("\n\n**\n")





for i in range(len(target_files)):
    wmd_dis(target_files[i],i)





fdf=open("wmd_form_ner.csv","w")
for i in range(192):
    fdf.write(os.path.basename(target_files[i])+'\n')
    for j in range(10):
        fdf.write(os.path.basename(sourceF[i][j])+",")
    fdf.write("\n")
fdf.close()





rd=pd.read_csv("wmd_form_ner.csv",header=None).values





wmd_tgt=[]
for i in range(len(rd)):
    tgt=rd[i][0][:7]
    a=[0]*3
    ct=0
    for j in range(1,4):
        if rd[i][j][:7]==tgt:
            ct+=1
    wmd_tgt.append([rd[i][0],ct])





ct=0
for i in wmd_tgt:
    if i[1]==1:
        ct+=1
print(ct)




fd33=open("number_of_source_in_T3.csv","w")
for i in range(len(wmd_tgt)):
    fd33.write(wmd_tgt[i][0]+","+str(wmd_tgt[i][1])+"\n")
fd33.close()





rd=pd.read_csv("result_from_ner_withpath-basename.csv",header=None)





rd2=rd.values

ner_tgt=[]
for i in range(len(rd2)):
    tgt=rd2[i][0][:7]
    a=[0]*3
    ct=0
    for j in range(1,len(rd2[i])-1):
        print(i,j)
        if rd2[i][j][:7]==tgt:
            ct+=1
    ner_tgt.append([rd2[i][0],ct])



fd33=open("number_of_source_in_Top_10_fromNER.csv","w")
for i in range(len(ner_tgt)):
    fd33.write(ner_tgt[i][0]+","+str(ner_tgt[i][1])+"\n")
fd33.close()




dfn=pd.read_csv("result_from_ner_withpath.csv",header=None).values



df22=open("final_ner.csv","w")
for i in range(len(dfn)):
    for j in range(len(dfn[i])-1):
        df22.write(os.path.basename(dfn[i][j])+",")
    df22.write("\n")
df22.close()

