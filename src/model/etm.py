import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from src.data_manager.data_manager import DataManager
from src.utils.utils import Utils
from src.config.config import Config as cfg


class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, emsize, 
                    theta_act, embeddings, train_embeddings, enc_drop):
        super(ETM, self).__init__()
        
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        print(f"vocab_size: {vocab_size}")
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)
        self.theta_act = self.get_activation(theta_act)
        
        # Define word embedding matrix, or rho
        # QUESTION: what is rho?
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            num_embeddings, embed_dim = embeddings.size()
            rho = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)
            '''self.rho = embeddings.clone().float().to(cfg.DEVICE)'''
            self.rho = nn.Parameter(embeddings.clone().float().to(cfg.DEVICE), requires_grad=False)
        
        # Define the matrix containing the topic embeddings
        # QUESTION: why do we need this?
        self.alphas = nn.Linear(rho_size, num_topics, bias=False) # QUESTION: why configure like that?
        
        # Variational distribution of theta_{1:D} 
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size, t_hidden_size),
            self.theta_act,
            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
        )
        
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True) # QUESTION: why configure like this?
        self.log_sigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        
        
    def encode(self, bows):
        q_theta = self.q_theta(bows) 
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta) # QUESTION: what does drop do?
        
        mu_theta = self.mu_q_theta(q_theta)
        log_sigma_theta = self.log_sigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + log_sigma_theta - mu_theta.pow(2) - log_sigma_theta.exp(), dim=-1).mean()
        
        return mu_theta, log_sigma_theta, kl_theta
        
    def decode(self, theta, beta):
        res = torch.mm(theta, beta) # QUESTION: what does mm do?
        almost_zeros = torch.full_like(res, 1e-6) # why need almost_zeros? can we replace full_like by another function?
        results_without_zeros = res.add(almost_zeros)
        predictions = torch.log(results_without_zeros)
        return predictions
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else: # QUESTION: why need this else block?
            return mu
    
    # @TODO: read this function
    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta
        
    def get_theta(self, normalized_bows):
        mu_theta, log_sigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, log_sigma_theta)
        theta = F.softmax(z, dim=-1) # QUESTION: why dim=-1?
        return theta, kld_theta
        
    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        # QUESTION: what is kld_theta? why code like this?
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None
        
        beta = self.get_beta()
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1) # @TODO: explain this formula
        
        if aggregate: # QUESTION: what is aggregate? why if aggregate then we have to calculate the mean of recon_loss like this?
            recon_loss = recon_loss.mean() 
        
        return recon_loss, kld_theta
        
    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Activation function not supported, defaulting to tanh function...')
            act = nn.Tanh()
        return act 
        
    def get_optimizer(self, args):
        if args.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'asgd':
            optimizer = optim.ASGD(self.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
        else:
            print('Optimizer not supported, defaulting to vanilla SGD...')
            optimizer = optim.SGD(self.parameters(), lr=args.lr)
        self.optimizer = optimizer
        return optimizer
    
    def train_epoch(self, epoch, args, train_set):
        self.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        num_docs = torch.randperm(args.num_train_docs) # QUSETION: what does torch.randperm do?
        batch_indices = torch.split(num_docs, args.batch_size) # QUESTION: what is this?
        
        for idx, indices in enumerate(batch_indices):
            self.optimizer.zero_grad()
            self.zero_grad()
            
            data_batch = DataManager.get_batch(train_set, indices)
            normalized_data_batch = data_batch
            recon_loss, kld_theta = self.forward(data_batch, normalized_data_batch)
            total_loss = recon_loss + kld_theta
            total_loss.backward()
            
            # QUESTION: what does this code do?
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), args.clip)
            
            self.optimizer.step()
            
            acc_loss += torch.sum(recon_loss).item() # QUESTION: recon_loss and kld_theta are arrays?
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            
            cnt+= 1
            
            # Debug
            # Or logging purpose
            # @TODO: read to understand this code
            if idx % args.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2) 
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

                print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, idx, len(indices), self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        
        # Debug
        # Or logging purpose
        # @TODO: read to understand this code
        cur_loss = round(acc_loss / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)
        print('*'*100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        print('*'*100)
        
    def evaluate(self, epoch, args, source, train_set, vocabulary , test_half_1_set, test_half_2_set, tc=True, td=True):
        self.eval()
        with torch.no_grad():
            if source == 'val':
                indices = torch.split(torch.tensor(range(args.num_valid_docs)), args.eval_batch_size)
            else: 
                indices = torch.split(torch.tensor(range(args.num_test_docs)), args.eval_batch_size)

            ## get \beta here
            beta = self.get_beta()

            ### do dc and tc here
            acc_loss = 0
            cnt = 0
            indices_1 = torch.split(torch.tensor(range(args.num_test_half_1_docs)), args.eval_batch_size)
            for idx, indice in enumerate(indices_1):
                data_batch_1 = DataManager.get_batch(test_half_1_set, indice)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                theta, _ = self.get_theta(normalized_data_batch_1)
                ## get predition loss using second half
                data_batch_2 = DataManager.get_batch(test_half_2_set, indice)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                res = torch.mm(theta, beta)
                preds = torch.log(res)
                recon_loss = -(preds * data_batch_2).sum(1)
                loss = recon_loss / sums_2.squeeze()
                loss = np.nanmean(loss.numpy())
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_dc = round(math.exp(cur_loss), 1)
            print('*'*100)
            print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
            print('*'*100)
            if epoch == 99 and (tc or td):
                beta = beta.data.cpu().numpy()
                if tc:
                    print('Computing topic coherence...')
                    Utils.get_topic_coherence(beta, train_set, vocabulary)
                if td:
                    print('Computing topic diversity...')
                    Utils.get_topic_diversity(beta, 25)
            return ppl_dc