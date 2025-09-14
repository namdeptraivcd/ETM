import argparse
import torch
import numpy as np
from pathlib import Path
from src.data_manager.data_manager import DataManager
from src.model.etm import ETM
from src.trainer.trainer import Trainer
from src.config.config import Config as cfg 


def main():
    # Take input arguments
    parser = argparse.ArgumentParser()  
    
    # data and file related arguments
    parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
    parser.add_argument('--data_path', type=str, default='data/20ng', help='directory containing data')
    parser.add_argument('--emb_path', type=str, default='data/20ng_embeddings.txt', help='directory containing word embeddings') # QUESTION: what is 20ng_embeddings?
    parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training')
        
    # model related arguments
    parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
    parser.add_argument('--rho_size', type=int, default=768, help='dimension of rho')
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
    parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
    parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
    parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')
    
    # optimization related arguments
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train...150 for 20ng 100 for others')
    parser.add_argument('--mode', type=str, default='train', help='train or eval model')
    parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
    parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
    parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
    parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
    parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
    parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
    parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')
    
    # evaluation, visualization and logging related arguments
    parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
    parser.add_argument('--log_interval', type=int, default=2, help='when to log training')
    parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
    parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
    parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
    parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
    parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')
    
    args = parser.parse_args()
    
    # Set manual seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Get vocabulary, train set, validation set, test half 1 set, test half 2 set
    data_manager = DataManager()
    vocab, train_set, valid_set, test_half_1_set, test_half_2_set = data_manager.get_data(doc_terms_file_path=cfg.DOC_TERMS_FILE_PATH,
                                                                                          vocab_terms_file_path=cfg.VOCAB_TERMS_FILE_PATH)
    vocab_size = len(vocab)
    args.vocab_size = vocab_size
    args.num_train_docs = train_set.shape[0]
    args.num_valid_docs = valid_set.shape[0]
    args.num_test_docs = test_half_1_set.shape[0] + test_half_2_set.shape[0]
    args.num_test_half_1_docs = test_half_1_set.shape[0]
    args.num_test_half_2_docs = test_half_2_set.shape[0]
    
    # Read embeddings and auto-adjust rho_size
    embeddings = None
    embedding_method = "bert"  # Change to "fasttext" if you want to use FastText
    
    if not args.train_embeddings:
        embeddings = data_manager.read_embedding_matrix(vocab, cfg.DEVICE, load_trained=False, method=embedding_method)
        args.embeddings_dim = embeddings.size()
        
        # Auto-adjust rho_size based on embedding method
        if embedding_method == "bert":
            args.rho_size = cfg.BERT_DIM  # 768
            print(f"🔧 Using BERT embeddings: rho_size = {args.rho_size}")
        elif embedding_method == "fasttext":
            args.rho_size = cfg.FASTTEXT_DIM  # 300
            print(f"🔧 Using FastText embeddings: rho_size = {args.rho_size}")
    else:
        print(f"🔧 Training embeddings from scratch: rho_size = {args.rho_size}")
        
    print(f"📊 Final model config: vocab_size={len(vocab)}, num_topics={args.num_topics}, rho_size={args.rho_size}")
        
    # Define checkpoint
    '''if args.mode == "eval":
        ckpt = args.load_from
    else:
        # @TODO: simplify this code
        ckpt = Path.cwd().joinpath(args.save_path, 
        'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
        args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
            args.lr, args.batch_size, args.rho_size, args.train_embeddings))'''
    
    # Initialize model, optimizer
    model = ETM(args.num_topics, 
            args.vocab_size, 
            args.t_hidden_size, 
            args.rho_size, 
            args.emb_size, 
            args.theta_act, 
            embeddings, 
            args.train_embeddings, 
            args.enc_drop).to(cfg.DEVICE)

    optimizer = model.get_optimizer(args)
    
    if args.mode == "train":
        best_epoch = 0
        best_val_ppl = 1e9
        all_val_ppls = []
        
        for epoch in range(0, args.epochs):
            model.train_epoch(epoch, args, train_set)
            val_ppl = model.evaluate(epoch, args, "val", train_set, vocab, test_half_1_set, test_half_2_set)
            print("Validation score: ", val_ppl)
            
            if val_ppl < best_val_ppl:
                '''with open(ckpt, "wb") as f:
                    torch.save(model, f)
                best_epoch = epoch
                best_val_ppl = val_ppl'''
                
                # Debug
                print("val_ppl < best_val_ppl")
            else:  # QUESTION: what does this code do?
                lr = optimizer.param_groups[0]["lr"]
                if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                    optimizer.param_groups[0]['lr'] /= args.lr_factor

            '''if epoch % args.visualize_every == 0:
                model.visualize(args, vocabulary = vocab)'''
            all_val_ppls.append(val_ppl)
        '''with open(ckpt, 'rb') as f:
            model = torch.load(f)
        model = model.to(cfg.DEVICE)
        val_ppl = model.evaluate(args, 'val', train_set, vocab,  test_half_1_set, test_half_2_set)'''
    # Train or evaluate model
    
    
if __name__ == "__main__":
    main()