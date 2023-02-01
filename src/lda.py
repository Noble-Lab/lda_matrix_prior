
#Load some packages
import numpy as np
import numpy.random as rand
import scipy.special
import collections
import copy
import itertools
import os
import sys
import pandas as pd
import warnings

#Helper functions

#Returns the ln of the beta function of list-like alpha.
def betaln(alpha) :
    return(sum([scipy.special.gammaln(a) for a in alpha]) - scipy.special.gammaln(sum(alpha)))

#Define some topic to word matrices
#@xsi: the number of words per document on average, simulated from geometric
#@n_doc: the number of documents to generate
#@n_vocab: the number of words in the vocabulary
#@n_topic: number of topics to generate
#@beta: the concentration parameter for the topic-word matrix
#@alpha: the concentration parameter for the document-topic matrix
#@RETURNS: corpus, true_topics, true_doc_topic, true_topic_word
#Define some topic to word matrices
#@xsi: the number of words per document on average, simulated from geometric
#@n_doc: the number of documents to generate
#@n_vocab: the number of words in the vocabulary
#@n_topic: number of topics to generate
#@beta: the concentration parameter for the topic-word matrix
#@alpha: the concentration parameter for the document-topic matrix
#@RETURNS: corpus, true_topics, true_doc_topic, true_topic_word
def generate_corpus(xsi, n_doc, n_vocab, n_topic, beta, alpha, true_topic_word=None, true_doc_topic=None, verbose=False) :
    #Topic-word probabilities
    if true_topic_word is None :
        true_topic_word = rand.dirichlet([beta] * n_vocab, n_topic)

    #Document-topic probabilities 
    if true_doc_topic is None:
        true_doc_topic = rand.dirichlet([alpha] * n_topic, n_doc)

    #True topic assignments
    true_topics = []
    
    #Generate documents
    corpus = []
    for i in range(n_doc) :
        if verbose:
            print("______________________")
            print(f"On doc {i}")
            print("______________________")
        n_word = rand.poisson(xsi)
        doc_topic = np.random.choice(a=np.arange(0,n_topic), size=n_word, p=true_doc_topic[i])
        doc_topic_counts = pd.Series(doc_topic).value_counts()
        sim_words = []
        for j in range(n_topic) :
            try :
                size = doc_topic_counts[j]
                word_num = np.random.choice(a=np.arange(0, n_vocab), size=size, p=true_topic_word[j])
                sim_words.append(word_num)
                if verbose:
                    print(f"Doctopic {doc_topic_counts[j]}")
                    print(f"Wordnum {word_num}")
            except KeyError: 
                continue
        doc = np.concatenate(sim_words)
        corpus.append(np.array(doc))
        true_topics.append(doc_topic)
    return corpus, true_topics, true_doc_topic, true_topic_word



#Create a corpus in the format of Lou Gevirtzman's Implementation of LDA
#@xsi: the number of words per document
#@n_doc: the number of documents to generate
#@n_vocab: the number of words in the vocabulary
#@n_topic: number of topics to generate
#@beta: the concentration parameter for the topic-word matrix
#@alpha: the concentration parameter for the document-topic matrix
#@true_topic_word: The true topic word matrix can be supplied
#@corpus_object: a corpus list that can be set instead of generating a new one. You still need to set n_doc, n_vocab, xsi, and n_topic, file_name, but alpha and beta probably don't matter.
def generate_corpus_lou(xsi, n_doc, n_vocab, n_topic, beta, alpha, file_name, true_topic_word=None, corpus_object=None, verbose=False) :
    #Creates a new corpus if a corpus is not supplied
    if corpus_object == None :
        corpus, true_topics, true_doc_topic, true_topic_word = generate_corpus(xsi, n_doc, n_vocab, n_topic, beta, alpha, true_topic_word)
    else :
        if xsi is not None or n_doc is not None or n_topic is not None or beta is not None or alpha is not None :
            warnings.warn("All other parameters are ignored if corpus_object is not None")
        corpus = corpus_object
    if n_doc is None or n_vocab is None:
        raise ValueError("You need to still specify what n_doc and n_vocab is????")
    #Tables the corpus
    sparse = [collections.Counter(corpus[i]) for i in range(n_doc)]
    if verbose :
        for s in sparse :
            print(s)
    #Make sure we aren't overwriting another file
    if os.path.exists(file_name) :
        raise ValueError(f"In generate_corpus_lou:{file_name} already exists")
    fn = open(file_name, "w+")
    #Write the number of cells
    fn.write(f"{n_doc}\n")
    #Write the size of the vocabulary
    fn.write(f"{n_vocab}\n")
    #Write a filler line
    fn.write("1000\n")
    for i, doc in enumerate(sparse) :
        for key in sorted(doc.keys()) :
            #Everything in lou's code is indexed at 1
            if verbose: print(f"Doc: {i+1} Key: {key+1} Value: {doc[key]}\n")
            fn.write(f"{i + 1}\t{key + 1}\t{doc[key]}\n")
    fn.close()
    return sparse


#Inference
#@corpus: a list of lists. First a list of documents is required, each document consisting 
#    of "words", or integers between 0 and n_vocab
#@n_topic: number of topics to use in the LDA
#@n_iter: iterations of Gibb's sampling to use
#@alpha: a vector with one entry for each topic
#@beta: a vector with one entry for each word.
def lda(corpus, n_vocab, n_topic, alpha, beta, n_iter = 200) :
    
    #Some tests on alpha and beta
    assert len(alpha) == n_topic, "Alpha incorrect dimensions"
    assert len(beta) == n_vocab, "Beta incorrect dimensions"
    
    #Compute the number of documents
    n_doc = len(corpus)
    
    #Start with guessing randomly the topic assignments for each of the words.
    #v is the assignments of each of the words
    v = []
    for doc in corpus :
        doc_v = []
        for word in doc :
            doc_v.append(rand.randint(0,n_topic))
        v.append(np.array(doc_v))

    #Calculate the number of times a word is assigned to a topic
    ntv = np.zeros([n_topic, n_vocab])
    #Calculate the number of times a topic is present in a document         
    mdt = np.zeros([n_doc, n_topic])
    #Caluclate the number of times a topic comes up
    t = np.zeros(n_topic)

    #Initialize the count matrices described in the document
    for i in range(len(v)):
        for j in range(len(v[i])):
            topic = v[i][j]
            word = corpus[i][j]
            ntv[topic, word] += 1
            mdt[i, topic] += 1
            t[topic] += 1

    #Keep track of best iteration
    best_v = copy.deepcopy(v)
    best_ll = -99999999999999

    #Fun plot to keep track of best ll versus iteration
    best_ll_track = []

    #Do gibbs sampling
    for iteration in range(n_iter): #niter
        print(iteration)

        ll = 0
        #Calculate the log likelihood of current iteration
        for i in range(n_doc) :
            ll += betaln(mdt[i,] + alpha) - betaln(alpha)
        for i in range(n_topic) :
            ll += betaln(ntv[i,] + beta) - betaln(beta)

        #If current ll is better than the best, update.
        if ll > best_ll :
            print("Old ll", best_ll, "-> New ll", ll)
            best_ll = ll
            best_v = copy.deepcopy(v)

        best_ll_track.append(best_ll)

        #Iterate through all the words
        for i in range(len(v)) : #v
            for j in range(len(v[i])) : #v[i]
                topic = v[i][j]
                word = corpus[i][j]
                ntv[topic, word] -= 1
                mdt[i, topic] -= 1
                t[topic] -= 1
                #Find the probabilities of the new topic assignments
                probs = np.zeros(n_topic)
                for k in range(n_topic) :
                    probs[k] = (mdt[i,k] + alpha[k])*(ntv[k, word] + beta[word])/(t[k] + sum(beta))
                probs = probs/p_sum(probs)
                topic = np.where(rand.multinomial(1, probs) == 1)[0][0]
                v[i][j] = topic
                ntv[topic, word] += 1
                mdt[i, topic] += 1
                t[topic] += 1
    return(best_v, best_ll_track)

#Beta should be either a vector of the beta parameter or 
def lda_likelihood_topics(corpus, topics, alpha, beta, n_vocab, n_topic, matrix=False):
    if len(alpha) != n_topic:
            raise ValueError("Number of topics needs to equal length of alpha")
    if not matrix:
        if len(beta) != n_vocab:
            raise ValueError("Number of topics needs to equal length of beta")
    else:
        if beta.shape[0] != n_topic:
            raise ValueError("beta.shape[0] != n_topic and matrix=True")
        if beta.shape[1] != n_vocab:
            raise ValueError("beta.shape[1] != n_vocab and matrix=True")
            
    #Compute the number of documents
    n_doc = len(corpus)

    #Start with guessing randomly the topic assignments for each of the words.
    #v is the assignments of each of the words
    v = topics

    #Calculate the number of times a word is assigned to a topic
    ntv = np.zeros([n_topic, n_vocab])
    #Calculate the number of times a topic is present in a document         
    mdt = np.zeros([n_doc, n_topic])
    #Caluclate the number of times a topic comes up
    t = np.zeros(n_topic)

    #Initialize the count matrices described in the document
    for i in range(len(v)):
        for j in range(len(v[i])):
            topic = v[i][j]
            word = corpus[i][j]
            ntv[topic, word] += 1
            mdt[i, topic] += 1
            t[topic] += 1
    
    if not matrix:
        ll = 0
        #Calculate the log likelihood of current iteration
        for i in range(n_doc) :
            ll += betaln(mdt[i,] + alpha) - betaln(alpha)
        for i in range(n_topic) :
            ll += betaln(ntv[i,] + beta) - betaln(beta)
        return(ll)
    else:
        ll = 0
        #Calculate the log likelihood of current iteration
        for i in range(n_doc) :
            ll += betaln(mdt[i,] + alpha) - betaln(alpha)
        for i in range(n_topic) :
            ll += betaln(ntv[i,] + beta[i,]) - betaln(beta[i,])
        return(ll)
        

def lda_likelihood_matrix(document_topic_counts, topic_word_counts, alpha, beta, n_vocab, n_topic, matrix=False):
    if len(alpha) != document_topic_counts.shape[1]:
        raise ValueError("len(alpha) != document_topic_counts.shape[1]")
    if len(alpha) != n_topic:
        raise ValueError("len(alpha) != n_topic")
    if len(alpha) != n_topic:
            raise ValueError("Number of topics needs to equal length of alpha")
    if not matrix:
        if len(beta) != n_vocab:
            raise ValueError("len(beta) != n_vocab")
        if len(beta) != topic_word_counts.shape[1]:
            raise ValueError("len(beta) != document_topic_word.shape[1]")
    else:
        if beta.shape[0] != n_topic:
            raise ValueError("beta.shape[0] != n_topic and matrix=True")
        if beta.shape[1] != n_vocab:
            raise ValueError("beta.shape[1] != n_vocab and matrix=True")
    mdt = document_topic_counts
    ntv = topic_word_counts
    n_doc = document_topic_counts.shape[0]
    ll = 0
    
    if not matrix:
        #Calculate the log likelihood of current iteration
        for i in range(n_doc) :
            ll += betaln(mdt[i,] + alpha) - betaln(alpha)
        for i in range(n_topic) :
            ll += betaln(ntv[i,] + beta) - betaln(beta)
        return(ll)
    else: 
        #Calculate the log likelihood of current iteration
        for i in range(n_doc) :
            ll += betaln(mdt[i,] + alpha) - betaln(alpha)
        for i in range(n_topic) :
            ll += betaln(ntv[i,] + beta[i,]) - betaln(beta[i,])
        return(ll)
    

#Returns a list of columns to pick for each row.
#    Uses a naiive brute force approach because I couldn't figure out how to do
#    a cool dynamic programming way
#@pred_topic_word, true_topic_word are np.arrays of the matrices. I think that the columns are topics and rows are documents.
def naive_best_topic_match(pred_topic_word, true_topic_word) :
    n_topic = pred_topic_word.shape[0]
    dist_matrix = np.zeros([n_topic, n_topic])
    for i in range(n_topic) :
        for j in range(n_topic) :
            dist_matrix[i,j] = sum((pred_topic_word[i] - true_topic_word[j])**2)
    perms = list(itertools.permutations(range(n_topic)))
    
    best_dist = 99999999
    best_p = np.repeat([-1], n_topic)
    for p in perms :
        dist = 0
        for i in range(n_topic) :
            dist += dist_matrix[i, p[i]]
        if dist < best_dist :
            print("bestp", best_p, "p", p)
            best_dist = dist
            best_p = p
    print(dist_matrix)
    return(best_p)
    
#normalizes a matrix by row. Changes the original matrix. Returns nothing.
def normalize_matrix_by_row(mat) :
    for i in range(mat.shape[0]) :
        mat[i,] = mat[i,]/sum(mat[i,])

# Compute the document topic matrix

#Takes the best predicted topics and true topics then returns 
#    the topic word matrix and document topic matrix.
#Returns predicted doc_topic, predicted topic_word
def topics_to_matrices(best_v, corpus, n_topic = 4, n_vocab = 9) :
    n_doc = len(corpus)
    pred_doc_topics = []
    for i in range(n_doc) :
        pred_doc_topic = [collections.Counter(best_v[i])[j] for j in range(n_topic)]
        pred_doc_topic_norm = [pred_doc_topic[j]/sum(pred_doc_topic) for j in range(n_topic)] 
        pred_doc_topics.append(pred_doc_topic_norm)
    #Compute the topic word matrix
    pred_topic_word = np.zeros([n_topic, n_vocab])
    for i in range(n_doc) :
        for j in range(len(best_v[i])) :
            word = corpus[i][j]
            ptopic = best_v[i][j]
            pred_topic_word[ptopic, word] += 1
    normalize_matrix_by_row(pred_topic_word)
    return pred_doc_topics, pred_topic_word

#Takes the best topic mapping (best_v), the true topic_word matrix, and true_doc_topic matrix
#Returns the mse, and plots a histogram if histogram is True
def evaluate_preds(best_v, corpus, true_topic_word, true_doc_topic, n_topic, n_vocab, histogram = True) :
    n_doc = len(corpus)
    
    pred_doc_topic, pred_topic_word = topics_to_matrices(best_v, corpus, n_topic = n_topic, n_vocab = n_vocab)

    #Normalize matrices to be proportions
    normalize_matrix_by_row(pred_topic_word)

    #Find the best matching of predicted topics to true topics
    topic_map = naive_best_topic_match(pred_topic_word, true_topic_word)
    print(topic_map)
    print(np.argsort(topic_map))

    #Reorder the columns of the document-topic matrix 
    pred_doc_topic = np.array(pred_doc_topic)[:,np.argsort(topic_map)]

    pred_errors = np.zeros(n_doc)
    for i in range(n_doc) :
        pred_doc_topic[i]
        true_doc_topic[i]
        pred_errors[i] = sum((pred_doc_topic[i] - true_doc_topic[i])**2)

    random_errors = np.zeros(n_doc)
    for i in range(n_doc) :
        d = rand.dirichlet([4/n_topic] * n_topic )
        d.sort()
        random_errors[i] = sum((d - true_doc_topic[i])**2)

    constant_errors = np.zeros(n_doc)
    for i in range(n_doc) :
        constant_errors[i] = sum((.25 - true_doc_topic[i])**2)
    
    if(histogram) :
        plt.hist(pred_errors, bins=np.arange(0, 1, 0.02), fc=(0, 0, 1, 0.5), linewidth = 1, edgecolor='black')
        plt.hist(random_errors, bins=np.arange(0, 1, 0.02), fc=(0, 1, 0, 0.5), linewidth = 1, edgecolor='black')
        plt.hist(constant_errors, bins=np.arange(0, 1, 0.02), fc=(1, 0, 0, 0.5), linewidth = 1, edgecolor='black')
        plt.title("LDA Prediction Errors after MAP 500 Iterations", pad = 15)
        plt.xlabel("Predicted Topic Prop - True Topic Prop", labelpad = 15)
        plt.ylabel("Count", labelpad=15)
        plt.legend(["LDA", "Dirichlet", "Constant .25"])

    return(np.mean(pred_errors))

def array_to_csv(arr, file_name) :
    fh = open(file_name, "w+")
    for line in arr :
        fh.write(",".join(map(str, line)) + "\n")

def mse(arr1, arr2, norm="l2") :
    if len(arr1) != len(arr2) :
        raise ValueError("Length of arr1 and arr2 need to be equal")
    if norm == "l2": 
        ret = np.mean((arr1 - arr2)**2)
    elif norm == "l1":
        ret = np.mean(np.abs(arr1 - arr2))
    else:
        raise ValueError("norm must be 'l1' or 'l2'")
    return(ret)
        

"""
Function applies the greedy MSE to find column of arr2 that matches best with column 1 of arr1. Then continues.
    This function checks that the rows sum to 1.
arr1: A np array with columns to match
arr2: Another np array with columns to match
norm: Use l2 if you want l2 norm, l1 if you want l1 norm
"""
def greedy_mse_dt(arr1, arr2, verbose=False, norm="l2") :
    if not np.all(arr1.shape == arr2.shape):
        raise ValueError("Shape of arr1 and arr2 must be equal")
    for i in range(arr1.shape[0]):
        if not np.isclose(sum(arr1[i,:]), 1):
            raise ValueError("Rows of arr1 must sum to 1")
    for i in range(arr2.shape[0]):
        if not np.isclose(sum(arr2[i,:]), 1):
            raise ValueError("Rows of arr2 must sum to 1")
    unmatched_columns = np.arange(0, arr1.shape[1])
    #For every column in arr1
    total_mse = 0
    for i in range(arr1.shape[1]):
        best_col = -1
        best_j = -1
        best_mse = 99999999999
        for j, col in enumerate(unmatched_columns):
            cur_mse = mse(arr1[:, i], arr2[:, col], norm=norm)
            if cur_mse < best_mse:
                best_j = j
                best_col = col
                best_mse = cur_mse
        if verbose: print(f"For column {i} best match was {best_col}")
        total_mse += best_mse
        unmatched_columns = np.delete(unmatched_columns, best_j)
    return(total_mse/arr1.shape[1])

def greedy_mse_tw(arr1, arr2, verbose=False, norm="l2") :
    if not np.all(arr1.shape == arr2.shape):
        raise ValueError("Shape of arr1 and arr2 must be equal")
    for i in range(arr1.shape[0]):
        if not np.isclose(sum(arr1[i,:]), 1):
            raise ValueError("Rows of arr1 must sum to 1")
    for i in range(arr2.shape[0]):
        if not np.isclose(sum(arr2[i,:]), 1):
            raise ValueError("Rows of arr2 must sum to 1")
    unmatched_rows = np.arange(0, arr1.shape[0])
    #For every row in arr1
    total_mse = 0
    for i in range(arr1.shape[0]):
        best_row = -1
        best_j = -1
        best_mse = 99999999999
        for j, row in enumerate(unmatched_rows):
            cur_mse = mse(arr1[i, :], arr2[row, :], norm=norm)
            if cur_mse < best_mse:
                best_j = j
                best_row = row
                best_mse = cur_mse
        if verbose: print(f"For row{i} best match was {best_row}")
        total_mse += best_mse
        unmatched_rows = np.delete(unmatched_rows, best_j)
    return(total_mse/arr1.shape[0])
            
def greedy_mse_tw_order(arr1, arr2, verbose=False, norm="l2") :
    if not np.all(arr1.shape == arr2.shape):
        raise ValueError("Shape of arr1 and arr2 must be equal")
    for i in range(arr1.shape[0]):
        if not np.isclose(sum(arr1[i,:]), 1):
            raise ValueError("Rows of arr1 must sum to 1")
    for i in range(arr2.shape[0]):
        if not np.isclose(sum(arr2[i,:]), 1):
            raise ValueError("Rows of arr2 must sum to 1")
    unmatched_rows = np.arange(0, arr1.shape[0])
    #For every row in arr1
    total_mse = 0
    order = np.zeros(arr1.shape[0], dtype=int)
    for i in range(arr1.shape[0]):
        best_row = -1
        best_j = -1
        best_mse = 99999999999
        for j, row in enumerate(unmatched_rows):
            cur_mse = mse(arr1[i, :], arr2[row, :], norm=norm)
            if cur_mse < best_mse:
                best_j = j
                best_row = row
                best_mse = cur_mse
        if verbose: print(f"For row{i} best match was {best_row}")
        total_mse += best_mse
        unmatched_rows = np.delete(unmatched_rows, best_j)
        order[i] = best_row
    return(order)

def greedy_mse_dt_order(arr1, arr2, verbose=False, norm="l2") :
    if not np.all(arr1.shape == arr2.shape):
        raise ValueError("Shape of arr1 and arr2 must be equal")
    for i in range(arr1.shape[0]):
        if not np.isclose(sum(arr1[i,:]), 1):
            raise ValueError("Rows of arr1 must sum to 1")
    for i in range(arr2.shape[0]):
        if not np.isclose(sum(arr2[i,:]), 1):
            raise ValueError("Rows of arr2 must sum to 1")
    unmatched_columns = np.arange(0, arr1.shape[1])
    #For every column in arr1
    total_mse = 0
    order = np.zeros(arr1.shape[1], dtype=int)
    for i in range(arr1.shape[1]):
        best_col = -1
        best_j = -1
        best_mse = 99999999999
        for j, col in enumerate(unmatched_columns):
            cur_mse = mse(arr1[:, i], arr2[:, col], norm=norm)
            if cur_mse < best_mse:
                best_j = j
                best_col = col
                best_mse = cur_mse
        if verbose: print(f"For column {i} best match was {best_col}")
        total_mse += best_mse
        unmatched_columns = np.delete(unmatched_columns, best_j)
        order[i] = best_col
    return(order)

def read_bow_to_array(fn, sep="\t"):
    f = open(fn, "r")
    n_doc = int(f.readline())
    n_vocab = int(f.readline())
    arr = np.zeros([n_doc, n_vocab])
    _ = f.readline()
    lines = f.readlines()
    for line in lines:
        row = [int(x) for x in line.split(sep)]
        arr[row[0] - 1, row[1] - 1] = row[2]
    return(arr)
        


