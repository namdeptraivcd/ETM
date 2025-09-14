import torch
from pathlib import Path


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data paths
    STOPWORDS_FILE_PATH = "data/raw_data/stopwords.txt"
    OUTPUT_DIR = Path("data/preprocessed_data/")
    NYT_PATH = "data/raw_data/nyt.txt"
    
    # Updated paths for BoW (not TF-IDF)
    DOC_TERMS_FILE_PATH = "data/preprocessed_data/bow_doc_terms_matrix.mat"
    VOCAB_TERMS_FILE_PATH = "data/preprocessed_data/bow_vocab_terms_matrix.mat"
    
    # Embedding paths
    EMBEDDINGS_PATH = "data/preprocessed_data/embedding_matrix.npy"
    BERT_EMBEDDINGS_PATH = "data/pre_trained_embeddings/bert_embedding_matrix.npy"
    FASTTEXT_EMBEDDINGS_PATH = "data/pre_trained_embeddings/fasttext_embedding_matrix.npy"
    
    # Model dimensions based on embedding method
    FASTTEXT_DIM = 300
    BERT_DIM = 768
    
    # ETM hyperparameters (from original paper)
    DEFAULT_VOCAB_SIZE = 2000
    DEFAULT_NUM_TOPICS = 50
    DEFAULT_T_HIDDEN_SIZE = 800
    DEFAULT_THETA_ACT = 'relu'
    DEFAULT_ENC_DROP = 0.5