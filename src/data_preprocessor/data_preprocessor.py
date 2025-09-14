import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from scipy.io import savemat
from src.config.config import Config as cfg 


class DataPreprocessor:
    
    def __init__(self, max_df=0.7, min_df=100, train_ratio=0.85, test_ratio=0.10):
        self.max_df = max_df
        self.min_df = min_df
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = 1.00 - train_ratio - test_ratio
        
        self.stop_words = []
        self.vocab = []
        self.word2id = []
        self.id2word = []
        self.docs = []
        self.num_docs = 0
    
    def load_stopwords(self, stopwords_file_path=cfg.STOPWORDS_FILE_PATH):
        with open(stopwords_file_path, 'r') as f:
            self.stop_words = f.read().split("\n")
    
    def process_dataset(self, data_file_path, output_dir=cfg.OUTPUT_DIR):
        self.__load_documents(data_file_path)
        self.__build_vocabulary()
        self.__split_documents()
        self.__create_bow_representations(output_dir)
        
        # Debug
        print("Done process dataset!")
    
    def __load_documents(self, data_file_path):
        with open(data_file_path, 'r') as f:
            self.docs = f.readlines()
        self.num_docs = len(self.docs)
    
    def __build_vocabulary(self):
        count_vectorizer = CountVectorizer(min_df=self.min_df, max_df=self.max_df, stop_words=None) # QUESTION: why stop_words=None?
        cvz = count_vectorizer.fit_transform(self.docs).sign() # QUESTION: why sign()?
        
        sum_counts = cvz.sum(axis=0) # QUESTION: what is axis? why axis=0?
        vocab_size = sum_counts.shape[1]
        sum_counts_np = np.zeros(vocab_size, dtype=int)
        
        for v in range(vocab_size):
            sum_counts_np[v] = sum_counts[0, v] # QUESTION: why [0, v]?
            
        # Debug 
        print(f"Initial vocabulary size: {vocab_size}")
         
        word2id = dict([(w, count_vectorizer.vocabulary_.get(w)) for w in count_vectorizer.vocabulary_]) 
        id2word = dict([(v, k) for k, v in word2id.items()])
        
        # Sort vocabulary by frequency
        idx_sort = np.argsort(sum_counts_np) # QUESTION: what is argsort()?
        sorted_vocab = [id2word[idx_sort[i]] for i in range(vocab_size)]
        
        # Filter out stopwords
        sorted_vocab = [w for w in sorted_vocab if w not in self.stop_words]
        
        # Debug
        print(f"Vocabulary size after removing stopwords: {len(sorted_vocab)}")
        
        self.vocab = sorted_vocab
        self.word2id = dict([(w, i) for i, w in enumerate(self.vocab)])
        self.id2word = dict([(v, k) for k, v in self.word2id.items()])
    
    def __split_documents(self):
        train_size = int(np.floor(self.train_ratio * self.num_docs))
        test_size = int(np.floor(self.test_ratio * self.num_docs))
        valid_size = int(self.num_docs - train_size - test_size)
        
        # Create random permutation for shuffle splitting
        idx_permute = np.random.permutation(self.num_docs).astype(int)
        
        # Filter vocabulary so that it only include words in training set
        train_words = set()
        for i in range(train_size):
            train_words.update(w for w in self.docs[idx_permute[i]].split() if w in self.word2id)
        
        self.vocab = list(train_words)
        self.word2id = dict([(w, i) for i, w in enumerate(self.vocab)])
        self.id2word = dict([(v, k) for k, v in self.word2id.items()])
        
        # Debug
        print(f"Vocabulary size after removing words which do not in training set: {len(self.vocab)}")
        
        # Convert documents to ID tokens
        self.train_docs = self.__tokenize_documents(idx_permute[:train_size])
        self.test_docs = self.__tokenize_documents(idx_permute[train_size:train_size + test_size])
        self.valid_docs = self.__tokenize_documents(idx_permute[train_size + test_size:])
        
        # Debug
        print(f'Number of documents (train): {len(self.train_docs)} [should be {train_size}]')
        print(f'Number of documents (test): {len(self.test_docs)} [should be {test_size}]')
        print(f'Number of documents (valid): {len(self.valid_docs)} [should be {valid_size}]')
        
        # Clean up
        self.train_docs = self.__remove_empty_documents(self.train_docs)
        self.test_docs = self.__remove_empty_documents(self.test_docs)
        self.valid_docs = self.__remove_empty_documents(self.valid_docs)
        
        # Remove test documents with length = 1
        self.test_docs = [doc for doc in self.test_docs if len(doc) > 1]
        
        # Split test set into two halves for robust testing purpose
        self.test_docs_half_1 = []
        self.test_docs_half_2 = []
        for doc in self.test_docs:
            doc_half_1 = []
            doc_half_2 = []
            for i, w in enumerate(doc):
                if i <= len(doc) / 2.0 - 1:
                    doc_half_1.append(w)
                else:
                    doc_half_2.append(w)
            self.test_docs_half_1.append(doc_half_1)
            self.test_docs_half_2.append(doc_half_2)
   
    def __tokenize_documents(self, doc_indices):
        tokenized_docs = []
        for idx in doc_indices:
            tokenized_doc = []
            for w in self.docs[idx].split():
                if w in self.word2id:
                    tokenized_doc.append(self.word2id[w])
            tokenized_docs.append(tokenized_doc)
        return tokenized_docs
     
    def __remove_empty_documents(self, docs):
        return [doc for doc in docs if doc != []]
    
    # @TODO: find out how the following methods works
    def __create_bow_representations(self, output_dir):
        # Create word lists and document indices for each set
        datasets = {
            'tr': self.train_docs,
            'ts': self.test_docs,
            'ts_h1': self.test_docs_half_1,
            'ts_h2': self.test_docs_half_2,
            'va': self.valid_docs
        }
        
        for name, docs in datasets.items():
            print(f'Processing {name} set...')
            
            # Create word list and document indices
            words = self.__create_word_list(docs)
            doc_indices = self.__create_document_indices(docs)
            
            print(f'  len(words_{name}): {len(words)}')
            print(f'  len(unique_doc_indices_{name}): {len(np.unique(doc_indices))} [should be {len(docs)}]')
            
            # Create bow representation
            bow = self.__create_bow_matrix(doc_indices, words, len(docs), len(self.vocab))
            
            # Split into tokens and counts
            tokens, counts = self.__split_bow_matrix(bow, len(docs))
            
            # Save to files
            savemat(f'{output_dir}bow_{name}_tokens.mat', {'tokens': tokens}, do_compression=True)
            savemat(f'{output_dir}bow_{name}_counts.mat', {'counts': counts}, do_compression=True)
            
            # Clean up memory
            del words, doc_indices, bow, tokens, counts
        
        # Save vocabulary
        with open(f'{output_dir}vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)
        
        print('Bag-of-words representations saved successfully!')
    
    def __create_word_list(self, docs):
        return [word for doc in docs for word in doc]
    
    def __create_document_indices(self, docs):
        doc_indices = []
        for doc_idx, doc in enumerate(docs):
            doc_indices.extend([doc_idx] * len(doc))
        return doc_indices
    
    def __create_bow_matrix(self, doc_indices, words, n_docs, vocab_size):
        return sparse.coo_matrix(
            ([1] * len(doc_indices), (doc_indices, words)), 
            shape=(n_docs, vocab_size)
        ).tocsr()
    
    def __split_bow_matrix(self, bow_matrix, n_docs):
        tokens = [[w for w in bow_matrix[doc, :].indices] for doc in range(n_docs)]
        counts = [[c for c in bow_matrix[doc, :].data] for doc in range(n_docs)]
        return tokens, counts
    

# Example usage
'''
if __name__ == "__main__":
    # Create preprocessor instance
    preprocessor = DataPreprocessor(max_df=0.7, min_df=100)
    
    # Load stopwords and process documents
    preprocessor.load_stopwords()
    preprocessor.process_dataset(cfg.NYT_PATH)
'''