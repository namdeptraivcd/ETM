import torch
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from gensim.models.fasttext import FastText as FT_gensim
from src.config.config import Config as cfg


class DataManager:
    def __init__(self, preprocessed_data_dir="data/preprocessed_data"):
        self.preprocessed_data_dir = Path().cwd().joinpath(preprocessed_data_dir)
        
    @staticmethod
    def read_mat_file(key, file_path):
        data = loadmat(file_path)[key]
        return data
        
    @staticmethod
    def split_train_test_matrix(dataset):
        # QUESTION: why need splitting step here while DataPreprocessor has already did it?
        train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=1)
        test_half_1_set, test_half_2_set = train_test_split(test_set, test_size=0.5, random_state=1)
        train_set, valid_set = train_test_split(train_set, test_size=0.25, random_state=1)
        return train_set, valid_set, test_half_1_set, test_half_2_set
    
    @staticmethod
    def get_data(doc_terms_file_path=cfg.DOC_TERMS_FILE_PATH, vocab_terms_file_path=cfg.VOCAB_TERMS_FILE_PATH):
        doc_term_matrix = DataManager.read_mat_file("doc_terms_matrix", doc_terms_file_path)
        vocab_terms = DataManager.read_mat_file("vocab_terms_matrix", vocab_terms_file_path)
        vocab = [str(x[0]) for x in vocab_terms.ravel()]  # Flatten vocab
        train_set, valid_set, test_half_1_set, test_half_2_set = DataManager.split_train_test_matrix(doc_term_matrix)
        return vocab, train_set, valid_set, test_half_1_set, test_half_2_set
        
    @staticmethod
    def get_batch(doc_terms_matrix, indices):
        data_batch = doc_terms_matrix[indices, :]
        data_batch = torch.from_numpy(data_batch.toarray()).float().to(cfg.DEVICE)
        return data_batch
        
    # @TODO: read this function
    @staticmethod
    def read_embedding_matrix(vocab, device, load_trained=True, method='bert'):
        """
        read the embedding matrix for the vocabulary using FastText or BERT
        Args:
            vocab: vocabulary list
            device: torch device
            load_trained: whether to load pre-trained embeddings
            method: 'fasttext' or 'bert'
        """
        if method == 'fasttext':
            model_path = "data/pre_trained_embeddings/nyt_fasttext.model"
            embeddings_path = cfg.FASTTEXT_EMBEDDINGS_PATH
            emb_dim = cfg.FASTTEXT_DIM  # 300 for FastText
            
            if load_trained:
                embeddings_matrix = np.load(embeddings_path, allow_pickle=True)
            else:
                model_gensim = FT_gensim.load(str(model_path))
                embeddings_matrix = np.zeros(shape=(len(vocab), emb_dim))
                print("🔄 Extracting FastText embeddings for vocab...")
                vocab = np.array(vocab).ravel()
                for index, word in tqdm(enumerate(vocab)):
                    try:
                        vector = model_gensim.wv.get_vector(word)
                    except KeyError:
                        vector = np.random.normal(size=(emb_dim,))
                    embeddings_matrix[index] = vector
                print("✅ Done extracting FastText embeddings.")
                np.save(embeddings_path, embeddings_matrix)
                
            embeddings = torch.from_numpy(embeddings_matrix).to(device)
            return embeddings
            
        elif method == 'bert':
            embeddings_path = cfg.BERT_EMBEDDINGS_PATH
            emb_dim = cfg.BERT_DIM  # 768 for BERT
            
            if load_trained:
                try:
                    embeddings_matrix = np.load(embeddings_path, allow_pickle=True)
                    print(f"📁 Loaded BERT embeddings from {embeddings_path}")
                except FileNotFoundError:
                    print(f"⚠️  BERT embeddings not found at {embeddings_path}, generating new ones...")
                    load_trained = False
            
            if not load_trained:
                from transformers import AutoTokenizer, AutoModel
                bert_model_name = 'bert-base-uncased'
                tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
                model = AutoModel.from_pretrained(bert_model_name).to(device)
                model.eval()
                
                embeddings_matrix = np.zeros((len(vocab), emb_dim))
                print("🔄 Extracting BERT embeddings for vocab...")
                
                with torch.no_grad():
                    for idx, word in tqdm(enumerate(vocab), total=len(vocab)):
                        tokens = tokenizer.tokenize(word)
                        if not tokens:
                            embeddings_matrix[idx] = np.random.normal(size=(emb_dim,))
                            continue
                        input_ids = tokenizer.convert_tokens_to_ids(tokens)
                        input_ids = torch.tensor([input_ids]).to(device)
                        outputs = model(input_ids)
                        # Average the token embeddings
                        word_emb = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()
                        embeddings_matrix[idx] = word_emb
                        
                print("✅ Done extracting BERT embeddings.")
                np.save(embeddings_path, embeddings_matrix)
                
            embeddings = torch.from_numpy(embeddings_matrix).to(device)
            return embeddings
            
        else:
            raise ValueError("❌ Unknown embedding method: {}. Use 'fasttext' or 'bert'".format(method))