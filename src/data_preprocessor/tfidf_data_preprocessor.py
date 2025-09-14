import pickle
import numpy as np
from pathlib import Path
from scipy.io import savemat
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config.config import Config as cfg


class TfidfDataPreprocessor:
    def __init__(self, max_df=0.7, min_df=100, max_features=200000, ngram_range=(1, 2)): # QUESTION: what is ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features    
        self.ngram_range = ngram_range
        
        self.docs = []
        self.vocab = []
        self.vectorizer = None
        self.doc_terms_matrix = None
    
    def process_dataset(self, data_file_path, output_dir=cfg.OUTPUT_DIR, file_prefix="tfidf"):
        self.__load_documents(data_file_path)
        self.__create_tfidf_matrix()
        self.__extract_vocabulary()
        self.__save_processed_data(output_dir, file_prefix)
        
    def __load_documents(self, data_file_path):
        with open(data_file_path, 'r', encoding="utf-8") as f:
            self.docs = f.readlines()
            
        self.docs = [doc.strip() for doc in self.docs if doc.strip()]
        
    def __create_tfidf_matrix(self):
        # QUESTION: why need self.vectorizer but not vectorizer only?
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode", # QUESTION: what is it used for?
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            max_df=self.max_df, # Ignore terms in more than max_df fraction of docs
            min_df=self.min_df # Ignore terms in fewer than min_df documents
        )
        
        # Tfidf sparse matrix
        self.doc_terms_matrix = self.vectorizer.fit_transform(self.docs) 
    
    def __extract_vocabulary(self):
        self.vocab = self.vectorizer.get_feature_names_out().tolist()
        
        # Debug
        if len(self.vocab) > 0:
            sample_terms = self.vocab[:10] if len(self.vocab) >= 10 else self.vocab
            print(f"Sample terms: {sample_terms}")
        else:
            raise ValueError("Vocabulary size is 0!")
        
    def __save_processed_data(self, output_dir, file_prefix):
        # Save vocabulary as pickle file
        vocab_file_path = output_dir / "vocab.pkl"
        with open(vocab_file_path, 'wb') as f:
            pickle.dump(self.vocab, f)
        
        # Debug
        print(f"Vocabulary saved to: {vocab_file_path}")
        
        # Save TF-IDF matrix as Matlab file
        matrix_file_path = output_dir / f"{file_prefix}_doc_terms_matrix.mat"
        savemat(matrix_file_path, 
                {"doc_terms_matrix": self.doc_terms_matrix},
                do_compression=True)
        
        # Debug
        print(f"Document terms matrix saved to: {matrix_file_path}")
        
        # Save vocabulary terms as Matlab file
        vocab_terms_file_path = output_dir / f"{file_prefix}_vocab_terms_matrix.mat"
        savemat(vocab_terms_file_path,
                {"vocab_terms_matrix": np.array(self.vocab, dtype=object)}, # QUESTION: why type=object?
                do_compression=True)
        
        # Debug
        print(f'Vocab terms saved to: {vocab_terms_file_path  }')
 
# Example usage
if __name__ == "__main__":
    processor = TfidfDataPreprocessor(max_features=200000, ngram_range=(1, 2))
    processor.process_dataset('data/raw_data/nyt.txt')

    # Get information about processed data
    '''info = processor.get_matrix_info()
    vocab = processor.get_vocabulary()'''