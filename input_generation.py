
import nltk
import os
from nltk.tokenize import sent_tokenize
import numpy as np
import pickle
import time
import sys
import torch


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns




def cumulative(arr):
    for i in range(len(arr)-1):
        arr[i+1]+=arr[i]
    return arr

dlnd_path = "TAPNew/"
all_direc = [dlnd_path+direc+"/"+direc1 for direc in os.listdir(dlnd_path) if os.path.isdir(dlnd_path+direc) for direc1 in os.listdir(dlnd_path+direc)]


# for removing hidden files on macOS
for j in range(10):
    for i in all_direc:
        term=os.path.basename(i)
        if(term[0]=='.'):
            all_direc.remove(i)

for i in xrange(len(all_direc)):
    all_direc[i]+=("/")

source_files = [[direc+"source/"+file for file in os.listdir(direc+"source/") if file.endswith(".txt") and not file.startswith('.')] for direc in all_direc]
target_files = [[direc+"target/"+file for file in os.listdir(direc+"target/") if file.endswith(".txt") and not file.startswith('.')] for direc in all_direc]
source_docs = [[sent_tokenize(open(file,"r").read().decode("ascii","ignore")) for file in direc] for direc in source_files]
target_docs = [[sent_tokenize(open(file,"r").read().decode("ascii","ignore")) for file in direc] for direc in target_files]
all_sentences = [sent for direc in source_docs for doc in direc for sent in doc]+[sent for direc in target_docs for doc in direc for sent in doc]





# Infersent (to convert in word vectors)
infersent = torch.load("infersent/encoder/model_1024_attn.pickle")
infersent.set_glove_path("glove.840B.300d.txt")
print("Infersent started!!")
start=time.time()
infersent.build_vocab(all_sentences,tokenize=True)
all_sentence_vectors = infersent.encode(all_sentences,tokenize=True)
print("Infersent done!!")
print("Time taken: "+str(time.time()-start))




all_sentence_vectors = np.split(all_sentence_vectors,cumulative([sum([len(doc) for doc in direc]) for direc in source_docs]+[sum([len(doc) for doc in direc]) for direc in target_docs]))
source_sentence_vectors = all_sentence_vectors[:len(source_docs)]
target_sentence_vectors = all_sentence_vectors[len(source_docs):len(source_docs)+len(target_docs)]
source_docs_vectors = [np.split(source_sentence_vectors[i],cumulative([len(doc) for doc in source_docs[i]])) for i in range(len(source_sentence_vectors))]
target_docs_vectors = [np.split(target_sentence_vectors[i],cumulative([len(doc) for doc in target_docs[i]])) for i in range(len(target_sentence_vectors))]





final_target_files=[]
for i in target_files:
    for j in i:
        s=os.path.basename(j)
        final_target_files.append(s)
print("There are ",len(final_target_files)," target files")


pickle.dump([source_docs_vectors,target_docs_vectors,source_docs,target_docs,source_files,target_files],open(str(len(final_target_files))+"_embdding.pickle","wb"))







# SETDV 
import pickle
import numpy as np
from scipy.spatial.distance import cdist
import xml.etree.ElementTree as ET





def build_rdv(t,s):
    return np.concatenate([t,s,np.subtract(t,s),np.multiply(t,s)],axis=0)
def rdv(target_matrix,source_matrix,target_files,dir_n,doc_n):   
    match = np.argmin(cdist(target_matrix,source_matrix,metric="cosine"),axis=1)
    relative_doc_vector = np.vstack((build_rdv(target_matrix[i],source_matrix[match[i]]) for i in range(target_matrix.shape[0])))
    label = [(float(tag.attrib["SLNS"])) for tag in ET.parse(target_files[dir_n][doc_n][:-4]+".xml").findall("feature") if "SLNS" in tag.attrib.keys()][0]
    return [relative_doc_vector,label]



source_docs_vectors,target_docs_vectors,source_docs,target_docs,source_files,target_files=pickle.load(open(str((7041))+"_embdding.pickle","rb"))
relative_doc_vectors = [rdv(target_docs_vectors[i][j],np.vstack((source_docs_vectors[i][k] for k in range(len(source_docs_vectors[i])))),target_files,i,j) for i in range(len(target_docs_vectors)) for j in range(len(target_docs_vectors[i])-1)]
pickle.dump(relative_doc_vectors,open(str(len(final_target_files))+"_setdv.pickle","wb"))



