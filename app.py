from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import datetime

import gensim
from gensim.models import KeyedVectors
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Parameter, LayerNorm
from torch.autograd import Variable
import torch.jit as jit
import torch.nn.functional as F
import time
import os
import math
from tqdm import tqdm
import collections
from collections import namedtuple
import random

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

engine = create_engine("postgresql://ahmed:786110@127.0.0.1:5432/zafar")
db = scoped_session(sessionmaker(bind=engine))

# global variables
data = {}

@app.route("/")
def index():
    global data
    data = {}
    return render_template("index.html")


@app.route("/getMessage", methods=['POST'])
def getMessage():
    message = request.form.get('message')
    name = request.form.get('name')
    email = request.form.get('email')
    timestamp = datetime.datetime.now()

    db.execute("INSERT INTO message (time, name, message, email) VALUES (:timeI, :nameI, :messageI, :emailI)",{"timeI":timestamp, "nameI":name, "messageI":message, "emailI":email})

    db.commit()

    return redirect(url_for('index'))


@app.route("/generate")
def generate():
    global data
    data = {}
    return render_template("generate.html")


@app.route("/getKeyWords", methods=['POST'])
def getKeyWords():    
    key1 = request.form.get('key1')
    key2 = request.form.get('key2')
    key3 = request.form.get('key3')
    key4 = request.form.get('key4')
    key5 = request.form.get('key5')
    numParas = request.form.get('numParas')
    
    global data

    data['key1'] = key1
    data['key2'] = key2
    data['key3'] = key3
    data['key4'] = key4
    data['key5'] = key5

    data['numParas'] = [i for i in range(1, int(numParas) + 1)]

    return redirect(url_for('paragraphs'))


@app.route("/paragraphs")
def paragraphs():

    global data

    data['paras'] = []

    for i in range(len(data['numParas'])):
        data['paras'].append(evaluateAndShowAttention([data['key1'],data['key2'],data['key3'],data['key4'],data['key5']], method='beam_search', is_sample=True))

    # out1 = evaluateAndShowAttention([data['key1'],data['key2'],data['key3'],data['key4'],data['key5']], method='beam_search', is_sample=True)

    # out2 = evaluateAndShowAttention([data['key5'],data['key4'],data['key3'],data['key2'],data['key1']], method='beam_search', is_sample=True)

    # out3 = evaluateAndShowAttention([data['key1'],data['key3'],data['key2'],data['key5'],data['key4']], method='beam_search', is_sample=True)

    # out = [out1, out2, out3]

    print(data)
    print([data['key1'],data['key3'],data['key2'],data['key5'],data['key4']])
    # print(out3)
    return render_template("paragraphs.html",data=data)


@app.route("/getFeedback", methods=['POST'])
def getFeedback():
    global data
    feedback = {}
    coherency = 0
    relevance = 0
    grammar = 0

    for i in range(1, len(data['numParas']) + 1):
        coherency = request.form.get('p'+str(i)+'-rate_coherency')
        relevance = request.form.get('p'+str(i)+'-rate_relevance')
        grammar = request.form.get('p'+str(i)+'-rate_grammar')
        feedback['para'+str(i)] = {'c': (0 if coherency == None else int(coherency) ),'r': (0 if relevance == None else int(relevance) ),'g': (0 if grammar == None else int(grammar) )}

    print(feedback)
    print(data)

    return redirect(url_for('generate'))


if __name__ == '__main__':

    deviceName = "cpu"
    device = torch.device(deviceName)

    file_path = "modelFiles/data_big_W2V.txt"

    fvec = KeyedVectors.load_word2vec_format(file_path, binary=False)
    word_vec = fvec.vectors
    vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    vocab.extend(list(fvec.vocab.keys()))
    word_vec = np.concatenate((np.array([[0]*word_vec.shape[1]] * 4), word_vec))
    word_vec = torch.tensor(word_vec).float()
    del fvec

    print("total %d words" % len(word_vec))


    #save_folder = "C:\Users\Leo-N\OneDrive\Desktop\University\Kaavish\Webpage_Zafar\modelTrained"

    word_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_word = {i: ch for i, ch in enumerate(vocab)}

    # run for inference 10
    def beam_search(topics, num_chars, model, idx_to_word, word_to_idx, is_sample=False):
        output_idx = [1]
        topics = [word_to_idx[x] for x in topics]
        topics = torch.tensor(topics)
        topics = topics.reshape((1, topics.shape[0]))
    #     hidden = torch.zeros(num_layers, 1, hidden_dim)
    #     hidden = (torch.zeros(num_layers, 1, hidden_dim).to(device), torch.zeros(num_layers, 1, hidden_dim).to(device))
        hidden = model.init_hidden(batch_size=1)
        if use_gpu:
    #         hidden = hidden.cuda()
            adaptive_softmax.to(device)
            topics = topics.to(device)
            # Also change seq_length here
            seq_length = torch.tensor(50).reshape(1, 1).to(device)
        """1"""    
        coverage_vector = model.init_coverage_vector(topics.shape[0], topics.shape[1])
        attentions = torch.zeros(num_chars, topics.shape[1])
        X = torch.tensor(output_idx[-1]).reshape((1, 1)).to(device)
        output = torch.zeros(1, hidden_dim).to(device)
        log_prob, output, hidden, attn_weight, coverage_vector = model.inference(inputs=X, 
                                                                    topics=topics, 
                                                                    output=output, 
                                                                    hidden=hidden, 
                                                                    coverage_vector=coverage_vector, 
                                                                    seq_length=seq_length)
        log_prob = log_prob.cpu().detach().reshape(-1).numpy()
    #     print(log_prob[10])
        """2"""
        if is_sample:
            top_indices = np.random.choice(vocab_size, beam_size, replace=False, p=np.exp(log_prob))
        else:
            top_indices = np.argsort(-log_prob)
        """3"""
        beams = [(0.0, [idx_to_word[1]], idx_to_word[1], torch.zeros(1, topics.shape[1]), torch.ones(1, topics.shape[1]))]
        b = beams[0]
        beam_candidates = []
    #     print(attn_weight[0].cpu().data, coverage_vector)
    #     assert False
        for i in range(beam_size):
            word_idx = top_indices[i]
            beam_candidates.append((b[0]+log_prob[word_idx], b[1]+[idx_to_word[word_idx]], word_idx, torch.cat((b[3], attn_weight[0].cpu().data), 0), torch.cat((b[4], coverage_vector.cpu().data), 0), hidden, output.squeeze(0), coverage_vector))
        """4"""
        beam_candidates.sort(key = lambda x:x[0], reverse = True) # decreasing order
        beams = beam_candidates[:beam_size] # truncate to get new beams
        
        for xy in range(num_chars-1):
            beam_candidates = []
            for b in beams:
                """5"""
                X = torch.tensor(b[2]).reshape((1, 1)).to(device)
                """6"""
                log_prob, output, hidden, attn_weight, coverage_vector = model.inference(inputs=X, 
                                                                            topics=topics, 
                                                                            output=b[6], 
                                                                            hidden=b[5], 
                                                                            coverage_vector=b[7], 
                                                                            seq_length=seq_length)
                log_prob = log_prob.cpu().detach().reshape(-1).numpy()
                """8"""
                if is_sample:
                    top_indices = np.random.choice(vocab_size, beam_size, replace=False, p=np.exp(log_prob))
                else:
                    top_indices = np.argsort(-log_prob)
                """9"""
                for i in range(beam_size):
                    word_idx = top_indices[i]
                    beam_candidates.append((b[0]+log_prob[word_idx], b[1]+[idx_to_word[word_idx]], word_idx, torch.cat((b[3], attn_weight[0].cpu().data), 0), torch.cat((b[4], coverage_vector.cpu().data), 0), hidden, output.squeeze(0), coverage_vector))
            """10"""
            beam_candidates.sort(key = lambda x:x[0], reverse = True) # decreasing order
            beams = beam_candidates[:beam_size] # truncate to get new beams
        
        """11"""
        if '<EOS>' in beams[0][1]:
            first_eos = beams[0][1].index('<EOS>')
        else:
            first_eos = num_chars-1
        return(''.join(beams[0][1][:first_eos]), beams[0][1][:first_eos], beams[0][3][:first_eos].t(), beams[0][4][:first_eos])

    # run for inference 11
    # plt.switch_backend('agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    def evaluateAndShowAttention(input_sentence, method='beam_search', is_sample=False):
        num_chars = 100     # change this to set size of output
        if method == 'beam_search':
            _, output_words, attentions, coverage_vector = beam_search(input_sentence, num_chars, model, idx_to_word, word_to_idx, is_sample=is_sample)
        else:
            _, output_words, attentions, _ = predict_rnn(input_sentence, num_chars, model, idx_to_word, word_to_idx)
        output = " ".join(output_words)
        print('input =', ' '.join(input_sentence))
        #print('output =', ' '.join(output_words))
        return output
    #     n_digits = 3
    #     coverage_vector = torch.round(coverage_vector * 10**n_digits) / (10**n_digits)
    #     coverage_vector=np.round(coverage_vector, n_digits)
    #     print(coverage_vector.numpy())

        #showAttention(' '.join(input_sentence), output_words, attentions)
    # run for inference 12



    embedding_dim = 100     # depends on your Word2Vec model embedding size
            # hidden_dim = 512
    hidden_dim = 128
    lr = 1e-3 * 0.5
    momentum = 0.01

            ## set to 10 from 100 by me
    num_epoch = 10

    clip_value = 0.1
    use_gpu = True
            # use_gpu = False
    num_layers = 2
            # num_layers = 1
    bidirectional = False
            # batch_size = 32
    batch_size = 4
    num_keywords = 5            # change this if more keywords
    verbose = 1
    check_point = 5
    beam_size = 2
    is_sample = True
    vocab_size = len(vocab)
            # device = torch.device(deviceName)
    loss_function = nn.NLLLoss()

    adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(hidden_dim, len(vocab), cutoffs=[round(vocab_size / 20), 4*round(vocab_size / 20)])



    # run for inference 7
    class Attention(nn.Module):
        """Implements Bahdanau (MLP) attention"""
        
        def __init__(self, hidden_size, embed_size):
            super(Attention, self).__init__()
            
            self.Ua = nn.Linear(embed_size, hidden_size, bias=False)
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Linear(hidden_size, 1, bias=True)
            # to store attention scores
            self.alphas = None
            
        def forward(self, query, topics, coverage_vector):
            scores = []
            C_t = coverage_vector.clone()
            for i in range(topics.shape[1]):
                proj_key = self.Ua(topics[:, i, :])
                query = self.Wa(query)
                scores += [self.va(torch.tanh(query + proj_key)) * C_t[:, i:i+1]]
                
            # stack scores
            scores = torch.stack(scores, dim=1)
            scores = scores.squeeze(2)
    #         print(scores.shape)
            # turn scores to probabilities
            alphas = F.softmax(scores, dim=1)
            self.alphas = alphas
            
            # mt vector is the weighted sum of the topics
            mt = torch.bmm(alphas.unsqueeze(1), topics)
            mt = mt.squeeze(1)
            
            # mt shape: [batch x embed], alphas shape: [batch x num_keywords]
            return mt, alphas
    # run for inference 8
    class AttentionDecoder(nn.Module):
        def __init__(self, hidden_size, embed_size, num_layers, dropout=0.5):
            super(AttentionDecoder, self).__init__()
            
            self.hidden_size = hidden_size
            self.embed_size = embed_size
            self.num_layers = num_layers
            self.dropout = dropout
            
            # topic attention
            self.attention = Attention(hidden_size, embed_size)
            
            # lstm
            self.rnn = nn.LSTM(input_size=embed_size * 2, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            dropout=dropout)
            
        def forward(self, input, output, hidden, phi, topics, coverage_vector):
            # 1. calculate attention weight and mt
            mt, score = self.attention(output.squeeze(0), topics, coverage_vector)
            mt = mt.unsqueeze(1).permute(1, 0, 2)
            
            # 2. update coverge vector [batch x num_keywords]
            coverage_vector = coverage_vector - score / phi
            
            # 3. concat input and Tt, and feed into rnn 
            output, hidden = self.rnn(torch.cat([input, mt], dim=2), hidden)
            
            return output, hidden, score, coverage_vector

    # run for inference 9
    LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

    class MTALSTM(nn.Module):
        def __init__(self, hidden_dim, embed_dim, num_keywords, num_layers, weight,
                    num_labels, bidirectional, dropout=0.5, **kwargs):
            super(MTALSTM, self).__init__(**kwargs)
            self.hidden_dim = hidden_dim
            self.embed_dim = embed_dim
            self.num_layers = num_layers
            self.num_labels = num_labels
            self.bidirectional = bidirectional
            if num_layers <= 1:
                self.dropout = 0
            else:
                self.dropout = dropout
            self.embedding = nn.Embedding.from_pretrained(weight)
            self.embedding.weight.requires_grad = False
            self.Uf = nn.Linear(embed_dim * num_keywords, num_keywords, bias=False)
            
            # attention decoder
            self.decoder = AttentionDecoder(hidden_size=hidden_dim, 
                                            embed_size=embed_dim, 
                                            num_layers=num_layers, 
                                            dropout=dropout)
            
            # adaptive softmax
            self.adaptiveSoftmax = nn.AdaptiveLogSoftmaxWithLoss(hidden_dim, 
                                                                num_labels, 
                                                                cutoffs=[round(num_labels / 20), 4*round(num_labels / 20)])
        
        def forward(self, inputs, topics, output, hidden=None, mask=None, target=None, coverage_vector=None, seq_length=None):
            embeddings = self.embedding(inputs)
            topics_embed = self.embedding(topics)
            ''' calculate phi [batch x num_keywords] '''
            phi = None
            phi = torch.sum(mask, dim=1, keepdim=True) * torch.sigmoid(self.Uf(topics_embed.reshape(topics_embed.shape[0], -1).float()))
            
            # loop through sequence
            inputs = embeddings.permute([1, 0, 2]).unbind(0)
            output_states = []
            attn_weight = []
            for i in range(len(inputs)):
                output, hidden, score, coverage_vector = self.decoder(input=inputs[i].unsqueeze(0), 
                                                                            output=output, 
                                                                            hidden=hidden, 
                                                                            phi=phi, 
                                                                            topics=topics_embed, 
                                                                            coverage_vector=coverage_vector) # [seq_len x batch x embed_size]
                output_states += [output]
                attn_weight += [score]
                
            output_states = torch.stack(output_states)
            attn_weight = torch.stack(attn_weight)
            
            # calculate loss py adaptiveSoftmax
            outputs = self.adaptiveSoftmax(output_states.reshape(-1, output_states.shape[-1]), target.t().reshape((-1,)))
            
            return outputs, output_states, hidden, attn_weight, coverage_vector
        
        def inference(self, inputs, topics, output, hidden=None, mask=None, coverage_vector=None, seq_length=None):
            embeddings = self.embedding(inputs)
            topics_embed = self.embedding(topics)
        
            phi = None
            phi = seq_length.float() * torch.sigmoid(self.Uf(topics_embed.reshape(topics_embed.shape[0], -1).float()))
            
            queries = embeddings.permute([1, 0, 2])[-1].unsqueeze(0)
            
            inputs = queries.permute([1, 0, 2]).unbind(0)
            output_states = []
            attn_weight = []
            for i in range(len(inputs)):
                output, hidden, score, coverage_vector = self.decoder(input=inputs[i].unsqueeze(0), 
                                                                            output=output, 
                                                                            hidden=hidden, 
                                                                            phi=phi, 
                                                                            topics=topics_embed, 
                                                                            coverage_vector=coverage_vector) # [seq_len x batch x embed_size]
                output_states += [output]
                attn_weight += [score]
                
            output_states = torch.stack(output_states)
            attn_weight = torch.stack(attn_weight)
            
            outputs = self.adaptiveSoftmax.log_prob(output_states.reshape(-1, output_states.shape[-1]))
            return outputs, output_states, hidden, attn_weight, coverage_vector
        
        def init_hidden(self, batch_size):
    #         hidden = torch.zeros(num_layers, batch_size, hidden_dim)
    #         hidden = LSTMState(torch.zeros(batch_size, hidden_dim).to(device), torch.zeros(batch_size, hidden_dim).to(device))
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device), 
                    torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
            return hidden
        
        def init_coverage_vector(self, batch_size, num_keywords):
    #         self.coverage_vector = torch.ones([batch_size, num_keywords]).to(device)
            return torch.ones([batch_size, num_keywords]).to(device)
    #         print(self.coverage_vector)


    # run for inference 11.5
    # run for inference 12

    model = MTALSTM(hidden_dim=hidden_dim, embed_dim=embedding_dim, num_keywords=num_keywords, num_layers=num_layers, num_labels=len(vocab), weight=word_vec, bidirectional=bidirectional)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # uncommented by me 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #from keras_radam import RAdam
    #optimizer = RAdam(model.parameters(), lr=lr)


    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2, min_lr=1e-7, verbose=True)
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)
    #if use_gpu:
    #     model = nn.DataParallel(model)
    #     model = model.to(device)
    #  model = model.to('cuda:0')
    #  print("Dump to cuda")
    #else:

    model = model.to(device)


    save_folder = ""
    # run for inference 13
    version_num = 1
    # Type = 'best'
    Type = 'trainable'
    model_check_point = 'modelFiles/model_%s_%d.pk' % (Type, version_num)
    optim_check_point = 'modelFiles/optim_%s_%d.pkl' % (Type, version_num)
    loss_check_point = 'modelFiles/loss_%s_%d.pkl' % (Type, version_num)
    epoch_check_point = 'modelFiles/epoch_%s_%d.pkl' % (Type, version_num)
    bleu_check_point = 'modelFiles/bleu_%s_%d.pkl' % (Type, version_num)
    loss_values = []
    epoch_values = []
    bleu_values = []
    #if os.path.isfile(model_check_point):
    if True:
        print('Loading previous status (ver.%d)...' % version_num)
        model.load_state_dict(torch.load(model_check_point, map_location='cpu'))
        model = model.to(device)
        optimizer.load_state_dict(torch.load(optim_check_point, map_location='cpu'))
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2, min_lr=1e-7, verbose=True)
        loss_values = torch.load(loss_check_point)
        epoch_values = torch.load(epoch_check_point)
        bleu_values = torch.load(bleu_check_point)
        print('Load successfully')
    #else:
    #    print("ver.%d doesn't exist" % version_num)


    #run for inference 14
    #out = evaluateAndShowAttention(['ایک', 'پاکستان', 'بھارت', 'کشمیر', 'قبضہ'], method='beam_search', is_sample=True)
    #print("output =", out)

  

    app.run(debug=True)