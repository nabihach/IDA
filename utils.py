from preprocess import *

import os

import math
import datetime
import time

import torch.nn as nn
import torch.nn.functional as F

FLOAT_MIN = -1e+20

USE_CUDA = torch.cuda.is_available()

MLP_FILE = "mlp.pt"
ENC_1_FILE = "enc_1.pt"
ENC_2_FILE = "enc_2.pt"
I2W_FILE = "i2w.dict"
W2I_FILE = "w2i.dict"
W2C_FILE = "w2c.dict"
INF_FILE = "info.dat"
TEMP_DIR = "current_model/"
DATA_DIR = "data/"
MODEL_DIR = "trained_models/"
TRAIN_FILE = "train.dat"
VAL_1_FILE = "val_1.dat"
VAL_2_FILE = "val_2.dat"
SCORE_FILE = "scores.dat"
LOG_FILE = "log.txt"
glove_path = "/home/nasghar/glove.840B.300d.txt"



def load_glove_embeddings(word2idx, embedding_dim=50):
    with open(glove_path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split(" ")
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    logits_flat = logits.view(-1, logits.size(-1))
    
    testing_output_probs = F.softmax(logits_flat, dim=1)


    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()

    loss = losses.sum() / length.float().sum()
    return loss




def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))



def init_save_pretrained_model(args, model_dir):
    t = datetime.datetime.now()

    log_file = open(model_dir + LOG_FILE, 'a')
    arg_dict = vars(args)
    log_file.write("="*20+"\nResuming Training with Parameters:\n")
    for (arg, val) in arg_dict.items():
        log_file.write(str(arg) + ": " + str(val) + '\n')
    log_file.write("\n")
    log_file.close()

    return 


def save_logs(logs, path):
    log_file = open(path + LOG_FILE, 'a')
    try:
        for log in logs:
            log_file.write(str(log))
    except TypeError:
        log_file.write(str(logs))
    log_file.write("\n")
    log_file.close()

def save_scores(scores, path):
    score_file = open(path + SCORE_FILE, 'a')
    score_file.write(str(scores))
    score_file.write("\n")
    score_file.close()

def masked_softmax2(vec, mask, dim=1):
    fmin = FLOAT_MIN * (1-mask.float())#.float()
    return F.softmax(vec + fmin, dim=dim)
