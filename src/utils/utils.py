import torch 
import numpy as np
    

# @TODO: read to understand this code
class Utils:
    def __init__(self):
        pass

    @staticmethod
    def get_topic_diversity(beta, topk):
        num_topics = beta.shape[0]
        list_w = np.zeros((num_topics, topk))
        for k in range(num_topics):
            idx = beta[k,:].argsort()[-topk:][::-1]
            list_w[k,:] = idx
        n_unique = len(np.unique(list_w))
        TD = n_unique / (topk * num_topics)
        print('Topic diveristy is: {}'.format(TD))

    @staticmethod
    def get_document_frequency(data, wi, wj=None):
        if wj is None:
            D_wi = 0
            for l in range(data.shape[0]):
                doc = data[l].toarray().squeeze()
                if doc.shape == () or doc.size == 0:
                    continue
                if doc[wi] > 0:
                    D_wi += 1
            return D_wi
        D_wj = 0
        D_wi_wj = 0
        for l in range(data.shape[0]):
            doc = data[l].toarray().squeeze()
            if doc.shape == () or doc.size == 0:
                continue
            if doc[wj] > 0:
                D_wj += 1
                if doc[wi] > 0:
                    D_wi_wj += 1
        return D_wj, D_wi_wj 

    @staticmethod
    def get_topic_coherence(beta, data, vocab):
        D = data.shape[0] # number of docs...data is list of documents
        print('D: ', D)
        TC = []
        NPMI = []
        num_topics = len(beta)
        for k in range(num_topics):
            print('k: {}/{}'.format(k, num_topics))
            top_10 = list(beta[k].argsort()[-11:][::-1])
            top_words = [vocab[a] for a in top_10]
            TC_k = 0
            NPMI_k = 0
            counter = 0
            for i, word in enumerate(top_10):
                D_wi = Utils.get_document_frequency(data, word)
                j = i + 1
                tmp_tc = 0
                tmp_npmi = 0
                while j < len(top_10) and j > i:
                    D_wj, D_wi_wj = Utils.get_document_frequency(data, word, top_10[j])
                    # UMass TC
                    '''if D_wi_wj == 0:
                        f_wi_wj = -1
                    else:
                        f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )'''
                    if D_wi_wj == 0:
                        f_wi_wj = np.log((1) / (D_wj))
                    else:
                        f_wi_wj = np.log((D_wi_wj + 1) / (D_wj))
                    tmp_tc += f_wi_wj
                    # NPMI
                    if D_wi_wj == 0:
                        npmi = 0
                    else:
                        p_wi = D_wi / D
                        p_wj = D_wj / D
                        p_wi_wj = D_wi_wj / D
                        pmi = np.log(p_wi_wj / (p_wi * p_wj))
                        npmi = pmi / (-np.log(p_wi_wj))
                    tmp_npmi += npmi
                    j += 1
                    counter += 1
                TC_k += tmp_tc
                NPMI_k += tmp_npmi
            TC.append(TC_k)
            NPMI.append(NPMI_k)
        print('counter: ', counter)
        print('num topics: ', len(TC))
        TC_score = np.mean(TC) / counter
        NPMI_score = np.mean(NPMI) / counter
        print('Topic coherence (UMass-like) is: {}'.format(TC_score))
        print('Topic coherence (NPMI) is: {}'.format(NPMI_score))
        
    @staticmethod
    def nearest_neighbors(model, word):
        nearest_neighbors = model.wv.most_similar(word, topn=20)
        nearest_neighbors = [comp[0] for comp in nearest_neighbors]
        return nearest_neighbors