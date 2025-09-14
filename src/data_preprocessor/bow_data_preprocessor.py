import pickle
import numpy as np
from pathlib import Path
from scipy.io import savemat
from sklearn.feature_extraction.text import CountVectorizer
from src.config.config import Config as cfg


class BowDataPreprocessor:
    def __init__(self, max_df=0.9, min_df=20, max_features=2000, ngram_range=(1, 1)):
        """
        Bag-of-Words preprocessor for ETM (following original paper settings)
        
        Args:
            max_df: Ignore terms that appear in more than max_df fraction of docs
            min_df: Ignore terms that appear in fewer than min_df documents  
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Only use unigrams (1,1) as in original ETM
        """
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features    
        self.ngram_range = ngram_range
        
        self.docs = []
        self.vocab = []
        self.vectorizer = None
        self.doc_terms_matrix = None
    
    def process_dataset(self, data_file_path, output_dir=cfg.OUTPUT_DIR, file_prefix="bow"):
        self.__load_documents(data_file_path)
        self.__create_bow_matrix()
        self.__extract_vocabulary()
        self.__save_processed_data(output_dir, file_prefix)
        print(f"✅ Processed {len(self.docs)} documents with vocabulary size: {len(self.vocab)}")
        
    def __load_documents(self, data_file_path):
        with open(data_file_path, 'r', encoding="utf-8") as f:
            self.docs = f.readlines()
            
        self.docs = [doc.strip() for doc in self.docs if doc.strip()]
        print(f"📚 Loaded {len(self.docs)} documents")
        
    def __create_bow_matrix(self):
        # Use CountVectorizer for Bag-of-Words (not TF-IDF)
        self.vectorizer = CountVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=self.ngram_range,  # Only unigrams for ETM
            max_features=self.max_features,
            max_df=self.max_df,  # Remove very frequent words
            min_df=self.min_df,  # Remove very rare words
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic words, min 2 chars
        )
        
        # Count matrix (Bag-of-Words)
        self.doc_terms_matrix = self.vectorizer.fit_transform(self.docs) 
        print(f"📊 Created BoW matrix: {self.doc_terms_matrix.shape}")
    
    def __extract_vocabulary(self):
        self.vocab = self.vectorizer.get_feature_names_out().tolist()
        
        # Debug
        if len(self.vocab) > 0:
            sample_terms = self.vocab[:10] if len(self.vocab) >= 10 else self.vocab
            print(f"📝 Sample vocab: {sample_terms}")
            print(f"📈 Vocabulary size: {len(self.vocab)}")
        else:
            raise ValueError("❌ Vocabulary size is 0! Check your min_df/max_df settings.")
        
    def __save_processed_data(self, output_dir, file_prefix):
        # Save vocabulary as pickle file
        vocab_file_path = output_dir / "vocab.pkl"
        with open(vocab_file_path, 'wb') as f:
            pickle.dump(self.vocab, f)
        print(f"💾 Vocabulary saved to: {vocab_file_path}")
        
        # Save BoW matrix as Matlab file
        matrix_file_path = output_dir / f"{file_prefix}_doc_terms_matrix.mat"
        savemat(matrix_file_path, 
                {"doc_terms_matrix": self.doc_terms_matrix},
                do_compression=True)
        print(f"💾 Document terms matrix saved to: {matrix_file_path}")
        
        # Save vocabulary terms as Matlab file
        vocab_terms_file_path = output_dir / f"{file_prefix}_vocab_terms_matrix.mat"
        savemat(vocab_terms_file_path,
                {"vocab_terms_matrix": np.array(self.vocab, dtype=object)},
                do_compression=True)
        print(f"💾 Vocab terms saved to: {vocab_terms_file_path}")
 
# Example usage
if __name__ == "__main__":
    # ETM paper settings: smaller vocab, less aggressive filtering
    processor = BowDataPreprocessor(
        max_df=0.9,      # Keep more frequent words
        min_df=20,       # Less aggressive rare word removal  
        max_features=2000,  # Smaller vocab like ETM paper
        ngram_range=(1, 1)  # Only unigrams
    )
    processor.process_dataset('data/raw_data/nyt.txt')
