# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import UsedFunction as huft
import numpy as np
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as SRP
from sklearn.decomposition import LatentDirichletAllocation as LDA,NMF,TruncatedSVD as SVD
from scipy.sparse import csr_matrix
from time import time

class InputFileNotExists(Exception):
    pass
class DimReductionNotRecognized(Exception):
    pass
#Argument Setting
desc="Generate n Topic(s)"
parser=argparse.ArgumentParser(description=desc)
parser.add_argument("Clean_Tweets_Location")
parser.add_argument("n_topics")
parser.add_argument("UseKernel")
parser.add_argument("Dim_reduction")
parser.add_argument("Output_Folder")
parser.add_argument("Output_File")
parser.add_argument("Time_File")
args=parser.parse_args()

time_file=args.Time_File
dimred=args.Dim_reduction.lower()
UseKernel=bool(eval(args.UseKernel))
if dimred not in ['grp','svd']: 
    raise DimReductionNotRecognized('{} not recognized'.format(dimred))
if not os.path.exists(args.Clean_Tweets_Location):
    raise InputFileNotExists('Filenya gaada')
if not os.path.exists(args.Output_Folder):
    os.makedirs(args.Output_Folder)
hasil = open('{}/{}'.format(args.Output_Folder,args.Output_File), 'w')
n_topics=eval(args.n_topics)
with open(args.Clean_Tweets_Location,'rb') as handle:
    tfidf,tfidf_terms=pickle.load(handle)
komponen=5
if dimred == "grp":
    dr=GRP(n_components=komponen,random_state=11)
elif dimred == 'srp':
    if UseKernel:
        dr=SRP(n_components=komponen,random_state=11)
    else:
        dr=SRP(n_components=komponen,random_state=11,dense_output=True)
elif dimred == 'svd':
    dr=SVD(n_components=komponen,random_state=11)
if dimred in ['lda','nmf']:
    if dimred == 'lda':
        dr=LDA(n_components=n_topics,random_state=11).fit(tfidf)
    else:
        dr=NMF(n_components=n_topics,random_state=11).fit(tfidf)
    cntr=dr.components_
else:
    tfile=open("Time/{}".format(dimred.upper())+UseKernel*"-Kernel"+"/{}".format(time_file),'a+')
    timer=time()
    data=dr.fit_transform(tfidf)
    tfile.write("{}\n".format(time()-timer))
    tfile.close()
    cntr,u,d=huft.fcmeans(data.T,n_topics,m=1.5,error=0.005,maxiter=1000,UsingKernel=UseKernel)  
    if UseKernel:
        u = (u ** 1.5)*d
    temp = csr_matrix(np.ones((tfidf.shape[1],1)).dot(np.atleast_2d(u.sum(axis=1))).T)
    u = csr_matrix(u)
    cntr = np.asarray(u.dot(tfidf) / temp)
n_top_words = 10
for topic_idx, topic in enumerate(cntr):
    hasil.write("" + " ".join([tfidf_terms[i] for i in topic.argsort()[:-n_top_words - 1:-1]]) + "\n")
hasil.close()