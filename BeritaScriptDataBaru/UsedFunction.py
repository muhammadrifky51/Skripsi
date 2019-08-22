# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:35:15 2018

@author: Muhammad Rifky Y
"""
import os
import re
import itertools 
import spacy
import docx2txt
import math
import pandas as pd
import numpy as np
from spacy.lang.id import Indonesian
from pattern3.web import PDF
from bz2 import BZ2File as bz2
from html import unescape
from unidecode import unidecode
from nltk import sent_tokenize,ngrams
from datasketch import MinHash
from googletrans import Translator
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel,pairwise_distances
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


def adaAngka(s):
    return any(i.isdigit() for i in s)

def fixTags(t):
    getHashtags = re.compile(r"#(\w+)")
    pisahtags = re.compile(r'[A-Z][^A-Z]*')
    tagS = re.findall(getHashtags, t)
    for tag in tagS:
        if len(tag)>0:
            tg = tag[0].upper()+tag[1:]
            proper_words = []
            if adaAngka(tg):
                tag2 = re.split('(\d+)',tg)
                tag2 = [w for w in tag2 if len(w)>0]
                for w in tag2:
                    try:
                        int(w) # error if w not a number
                        proper_words.append(w)
                    except:
                        w = w[0].upper()+w[1:]
                        proper_words = proper_words+re.findall(pisahtags, w)
            else:
                proper_words = re.findall(pisahtags, tg)
            proper_words = ' '.join(proper_words)
            t = t.replace('#'+tag, proper_words)
    return t

def cleanText(T, fix={}, lemma=None, stops = set(), symbols_remove = False, min_charLen = 2, fixTag= True,user_remove=True):
    # lang & stopS only 2 options : 'en' atau 'id'
    # symbols ASCII atau alnum
    penerjemah=Translator()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',str(T)) #remove urls if any
    t = unescape(t) # html entities fix
    if fixTag:
        t = fixTags(t) # fix abcDef
    t = t.lower().strip() # lowercase
    t = unidecode(t)
    #t=re.sub(r'[m]*m','m',t)
    #t=re.sub(r'[a]*a','a',t)
    '''
    t=re.sub(r'([a-z])\1+',r'\1',t)
    t=re.sub(r'gogle','google',t)
    t=re.sub(r'[weak]*wk[weak]*','',t)
    t=re.sub(r'(he){2,}','',t)
    #t=re.sub(r'[bw]*aha','ha',t)
    t=re.sub(r'(ha){2,}','',t)
    '''
    t = ''.join(''.join(s)[:2] for _, s in itertools.groupby(t)) # remove repetition
    t = t.replace('\n', ' ').replace('\r', ' ')
    t = sent_tokenize(t) # sentence segmentation. String to list
    for i, K in enumerate(t):
        K=K.lower()
        #K=re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", K) #Delete Number
        K=re.sub(r"[0-9]*", " ", K) #Delete Number
        if user_remove:
            K = re.sub('@[^\s]+','',K) #remove user
        #K = re.sub('@[^\s]+','AT_USER',K)
        if symbols_remove:
            #K = re.sub(r'#[a-zA-Z0-9]*','',K)
            K = re.sub(r'[^\w]',' ',K)
        try:listKata, cleanList = lemma(K), []
        except:listKata,cleanList=K.split(),[]
        if len(listKata) is not 0:
            if not isinstance(listKata[0],str):
                for token in listKata:
                    if token.text in list(fix.keys()):
                        token = fix[token.text]
                    #try: 
                    if isinstance(token,str):token=lemma(token)
                    try:token=penerjemah.translate(token.text,dest='id').text
                    except:pass
                    if not isinstance(token,str):token=token.text
                    #try:token=stemmer.stem(token.text)
                    #except:token=stemmer.stem(token)
                    if not lemma:
                        try:
                            token = token.lemma_
                        except:
                            if len(token) is not 0:
                                token = lemma(token)[0].lemma_
                    if stops:
                        if len(token)>=min_charLen and token not in stops:
                            if token.lower() is not "pron":
                                cleanList.append(token)
                    else:
                        if len(token)>=min_charLen:
                            cleanList.append(token)
        t[i] = ' '.join(cleanList)
    return ' '.join(t) # Return kalimat lagi

def readBz2(file):
    with bz2(file, "r") as bzData:
        txt = []
        for line in bzData:
            try:
                txt.append(line.strip().decode('utf-8','replace'))
            except:
                pass
    return ' '.join(txt)

def crawlFiles(dPath,types=None): # dPath ='C:/Temp/', types = 'pdf'
    if types:
        return [dPath+f for f in os.listdir(dPath) if f.endswith('.'+types)]
    else:
        return [dPath+f for f in os.listdir(dPath)]

def LoadDocuments(dPath=None,types=None, file = None): # types = ['pdf','doc','docx','txt','bz2']
    Files, Docs = [], []
    if types:
        for tipe in types:
            Files += crawlFiles(dPath,tipe)
    if file:
        Files = [file]
    if not types and not file: # get all files regardless of their extensions
        Files += crawlFiles(dPath)
    for f in Files:
        if f[-3:].lower()=='pdf':
            try:
                Docs.append(PDF(f).string)
            except:
                print(('error reading{0}'.format(f)))
        elif f[-3:].lower() in ['txt', 'dic','py', 'ipynb']:
            try:
                df=open(f,"r",encoding="utf-8", errors='replace')
                Docs.append(df.readlines());df.close()
            except:
                print(('error reading{0}'.format(f)))
        elif f[-3:].lower()=='bz2':
            try:
                Docs.append(readBz2(f))
            except:
                print(('error reading{0}'.format(f)))
        elif f[-4:].lower()=='docx':
            try:
                Docs.append(docx2txt.process(f))
            except:
                print(('error reading{0}'.format(f)))
        elif f[-3:].lower()=='csv':
            Docs.append(pd.read_csv(f))
        else:
            print(('Unsupported format {0}'.format(f)))
    if file:
        Docs = Docs[0]
    return Docs, Files

def LoadStopWords(lang='en'):
    L = lang.lower().strip()
    if L == 'en' or L == 'english' or L == 'inggris':
        lemmatizer = spacy.load('en')
        stops =  set([t.strip() for t in LoadDocuments(file = 'data/stopwords_eng.txt')[0]])
    elif L == 'id' or L == 'indonesia' or L=='indonesian':
        lemmatizer = Indonesian()
        stops = set([t.strip() for t in LoadDocuments(file = 'data/stopwords_id.txt')[0]])
    else:
        print('Warning, language not recognized. Empty StopWords Given')
        stops = set(); lemmatizer = None
    return stops, lemmatizer

def fHasher(data, lsh): # Create MinHash objects
    minhashes = {}
    for c, i in enumerate(data):
      minhash = MinHash(num_perm=128)
      for d in ngrams(i, 3):
        minhash.update("".join(d).encode('utf-8'))
      lsh.insert(c, minhash)
      minhashes[c] = minhash
    return minhashes

def CalcJaccard(minhashes, T,lsh):
    hasil=[]
    for i in range(len(list(minhashes.keys()))):
      result = lsh.query(minhashes[i])
      if result not in hasil:hasil.append(result)
    return hasil

def slang2dict(f='data/slang.dic'):
    d={}
    fileloc=open(f,encoding='utf-8', errors ='ignore', mode='r')
    with fileloc as document:
        for line in document:
            doc=line.replace('\n',"")
            k,v=doc.split(':')
            d[k]=v
    return d

def _distance(data, centers):
    """
    Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.

    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    #print(type(pairwise_distances(data,centers)))
    return cdist(data, centers).T
    #return pairwise_distances(data, centers).T
    
def _kernel(data, centers):
    """
    Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.

    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    return rbf_kernel(data, centers, gamma=0.01).T

def fcmeans(data, c, m, error, maxiter, UsingKernel=False, init=None, seed=None):
    """
    Fuzzy c-means clustering algorithm [1].

    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    init : 2d array, size (S, c)
        Initial cluster centers. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Returns
    -------
    cntr : 2d array, size (S, c)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (c, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (c, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (c, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.


    Notes
    -----
    The algorithm implemented is from Ross et al. [1]_.

    Fuzzy C-Means has a known problem with high dimensionality datasets, where
    the majority of cluster centers are pulled into the overall center of
    gravity. If you are clustering data with very high dimensionality and
    encounter this issue, another clustering method may be required. For more
    information and the theory behind this, see Winkler et al. [2]_.

    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.

    .. [2] Winkler, R., Klawonn, F., & Kruse, R. Fuzzy c-means in high
           dimensional spaces. 2012. Contemporary Theory and Pragmatic
           Approaches in Fuzzy Computing Utilization, 1.
    """
    # Setup cntr
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        S = data.shape[0]
        cntr = np.random.rand(S, c)        
        init = cntr.copy()
            
    cntr = init
    # Setup u
    if UsingKernel:
        d = _kernel(data.T, cntr.T)
    else:
        d = _distance(data.T, cntr.T)
    d = np.fmax(d, np.finfo(np.float64).eps)
    if UsingKernel:
        u = (1-d) ** (- 1. / (m - 1))
    else:
        u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))
    
    
    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        upre = u.copy()
        [cntr, u, Jjm, d] = _fcmeans0(data, upre, c, m,d,UsingKernel)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - upre) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - upre)
    #fpc = _fp_coeff(u)

    #return cntr, u, u0, d, jm, p, fpc
    return cntr,u,d

def _fcmeans0(data, u_old, c, m,d,UsingKernel):
    """
    Single step in generic fuzzy c-means clustering algorithm.

    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.

    Parameters inherited from fcmeans()
    """
    # Normalizing, then eliminating any potential zero values.
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    if UsingKernel:
        cntr = (um*d).dot(data) / (np.ones((data.shape[1],1)).dot(np.atleast_2d((um*d).sum(axis=1))).T)
    else:
        cntr = um.dot(data) / (np.ones((data.shape[1],1)).dot(np.atleast_2d(um.sum(axis=1))).T)
	
    if UsingKernel:
        d = _kernel(data, cntr)
    else:
        d = _distance(data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)
	
    if UsingKernel:
        jm = (2 * um * (1-d)).sum()
        u = (1-d) ** (- 1. / (m - 1))
    else:
        jm = (um * d ** 2).sum()
        u = d ** (- 2. / (m - 1))

    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return cntr, u, jm, d

def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.

    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.

    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.

    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)