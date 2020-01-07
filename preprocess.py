

import torch
from torch.autograd import Variable

import re
import unicodedata
import random

import numpy as np

import pdb

MAX_SENTENCE_LENGTH = 50
MAX_SEQUENCE_LENGTH = MAX_SENTENCE_LENGTH + 2

SOS_INDEX = 0
EOS_INDEX = 1
UNK_INDEX = 2
BRK_INDEX = 3
PAD_INDEX = 4
ELP_INDEX = 5
NL_INDEX = 6

UNK = "<unk>"
BRK = "<brk>"
EOS = "<eos>"
SOS = "<sos>"
PAD = "<pad>"
ELP = "<elp>"
NL = "<nl>"

TOKENS = {r'\.\.\.': ' '+ELP+' '}
RESERVED_I2W = {SOS_INDEX: SOS, EOS_INDEX: EOS, UNK_INDEX: UNK, BRK_INDEX: BRK,
            PAD_INDEX: PAD, ELP_INDEX: ELP, NL_INDEX: NL}
RESERVED_W2I = dict((v,k) for k,v in RESERVED_I2W.items())

USE_CUDA = torch.cuda.is_available()
PUNCTUATION = ".!?,;:"
RMV_TOKENS = [EOS, SOS, PAD]

class WordDict(object):
    def __init__(self, dicts=None):
        if dicts == None:
            self._init_dicts()
        else:
            self.word2index, self.index2word, self.word2count, self.n_words = dicts

    def _init_dicts(self):
        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        self.index2word.update(RESERVED_I2W)
        self.word2index.update(RESERVED_W2I)

        self.n_words = len(RESERVED_I2W)  # number of words in the dictionary

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if not word in RESERVED_W2I:
            if not word in self.word2index:
                #print(word)
                self.word2index[word] = self.n_words
                #print(self.n_words)
                #input()
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def remove_unknowns(self, cutoff):
        # find unknown words
        unks = []
        for word, count in self.word2count.items():
            if word not in RESERVED_W2I and count <= cutoff:
                unks.append(word)

        # remove unknown words
        for word in unks:
            del self.index2word[self.word2index[word]]
            del self.word2index[word]
            del self.word2count[word]

        # reformat dictionaries so keys get shifted to correspond to removed words
        old_w2i = self.word2index
        old_w2c = self.word2count
        self._init_dicts()
        for word, index in old_w2i.items():
            if word not in RESERVED_W2I:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                if word in old_w2c:
                    self.word2count[word] = old_w2c[word]
                self.n_words += 1
        self.n_words = self.n_words

        return unks

    def to_indices(self, words):
        indices = []
        for word in words:
            if word in self.word2index:
                indices.append(self.word2index[word])
            else:
                indices.append(self.word2index[UNK])
        return indices

    def to_words(self, indices):
        words = []
        for index in indices:
            if index in self.index2word:
                words.append(self.index2word[index])
            else:
                words.append(UNK)
        return words


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize(s):
    s = unicode_to_ascii(s.lower().strip())
    for token, flag in TOKENS.items():
        s = re.sub(token, flag, s)
    s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"([!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?<>']+", r" ", s)
    #s = re.sub(r"[^a-zA-Z0123456789.!?<>:;$=/*%&@#|~\[\]'\-\+]+", r" ", s)
    return s

def separate(s, max_len=MAX_SENTENCE_LENGTH, separator=" "):
    return s.split(separator)[:max_len]

def get_pairs(lines):
    pairs = []
    msg = None
    resp = None
    addpair = False
    print("Collecting pairs of index lists.")
    for i in range(len(lines)):
        if not BRK in lines[i]:
            resp = lines[i]
            if addpair == True:
                pairs.append([msg, resp])
            msg = resp
            addpair = True
        else:
            addpair = False
    n_pairs = len(pairs)
    print(str(n_pairs) + " pairs of index lists collected.")
    return pairs

def get_validation_set(pairs, val_size):
    if val_size <= 0:
        return pairs, None
    n = len(pairs)
    indices = random.sample(range(n), val_size)
    train_set = []
    val_set = []
    for i in range(n):
        if not i in indices:
            train_set.append(pairs[i])
        else:
            val_set.append(pairs[i])
    print("Training and validation sets generated.")
    return train_set, val_set

#
#   Stage 3b: Insert SOS/EOS tokens and convert to indices
#

def tokenize(s, wd):
    return wd.to_indices(s) + [EOS_INDEX]

def tokenize_pairs(pairs, wd):
    tokenized_pairs = []
    for pair in pairs:
        tokenized_pairs.append([tokenize(s, wd) for s in pair])
    return tokenized_pairs


def pad_seq(seq, max_length):
    seq += [PAD_INDEX for i in range(max_length - len(seq))]
    return seq

def nli_batch_to_variable(text, hyp, label, sort=True):
    if sort:
        # Zip into pairs, sort by length (descending), unzip
        batch = sorted(zip(text, hyp, label), key=lambda p: len(p[0]), reverse=True)
        text, hyp, label = zip(*batch)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    text_lengths = [len(s) for s in text]
    text_padded = [pad_seq(s, max(text_lengths)) for s in text]

    hyp_lengths = [len(s) for s in hyp]
    hyp_padded = [pad_seq(s, max(hyp_lengths)) for s in hyp]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    text = np.array(text_padded, dtype='long').transpose(1,0)
    hyp = np.array(hyp_padded, dtype='long').transpose(1,0)
    label = np.array(label, dtype='long')

    return (text, text_lengths, hyp, hyp_lengths, label)

def get_nli_batch(batch_size, data):
    batches = []

    offset = 0
    total_data = data["length"]
    while offset < total_data:
        batch = [
            data["text"][ offset: min(offset + batch_size, total_data) ],
            data["hyp"][ offset: min(offset + batch_size, total_data) ],
            data["label"][ offset: min(offset + batch_size, total_data) ]
        ]
        batches.append( batch )
        offset += batch_size

    for i, batch in enumerate(batches):
        batches[i] = nli_batch_to_variable( batch[0], batch[1], batch[2] )

    return batches

def batch_pairs(inputs, targets, sort=True):
    if sort:
        # Zip into pairs, sort by length (descending), unzip
        seq_pairs = sorted(zip(inputs, targets), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)
    else:
        input_seqs, target_seqs = inputs, targets

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return (input_var, input_lengths, target_var, target_lengths)

def batches_by_length(batch_size, pairs, max_sequence_length):
    n_pairs = len(pairs)
    n_batches = int(n_pairs / batch_size)

    buckets = bucketize(pairs, max_sequence_length+1, key=lambda p: len(p[0]))

    batches = []
    done_all=False
    current_bucket = max_sequence_length
    current_index=0
    while not done_all:
        input_seqs = []
        target_seqs = []

        n_batch = batch_size

        done_batch=False
        while not done_batch:
            bucket = buckets[current_bucket]
            n_bucket = len(bucket)
            n_left = n_bucket - current_index
            n_current = min(n_batch, n_left)

            for i in range(current_index, current_index + n_current):
                input_seq, target_seq = bucket[i]
                input_seqs.append(input_seq)
                target_seqs.append(target_seq)

            current_index += n_current
            n_batch -= n_current

            if n_batch > 0:
                if current_bucket > 0:
                    current_bucket -= 1
                    current_index = 0
                else:
                    done_all=True
                    done_batch=True
            else:
                batch = batch_pairs(input_seqs, target_seqs)
                batches.append(batch)
                done_batch=True

    return batches



def bucketize(raw_list, max_len, key=len, shuffled=True):
    sorted_list = [[] for i in range(max_len)]
    for i in range(len(raw_list)):
        element = raw_list[i]
        val = key(element)
        sorted_list[val].append(element)

    if shuffled:
        for i in range(max_len):
            pairs_i = sorted_list[i]
            new_pairs = []
            n = len(pairs_i)
            p = np.random.permutation(n)
            for j in range(n):
                k = p[j]
                new_pairs.append(pairs_i[k])
            sorted_list[i] = new_pairs

    return sorted_list

def nli_random_batches(batch_size, data):
    n_batches = int(data["length"] / batch_size)
    p = np.random.permutation(data["length"])

    batches = []
    for i in range(n_batches):
        # Choose random pairs
        text = []
        hyp = []
        label = []
        for j in range(batch_size):
            idx = p[i*batch_size+j]
            text.append(data["text"][idx])
            hyp.append(data["hyp"][idx])
            label.append(data["label"][idx])

        batches.append(nli_batch_to_variable(text, hyp, label))

    return batches

def nli_batches(batch_size, data):
    n_batches = data["length"] #assumes batch_size = 1

    batches = []
    for i in range(n_batches):
        # Choose random pairs
        text = []
        hyp = []
        label = []
        for j in range(batch_size):
            idx = i+j
            data["text"][idx] = [x for x in data["text"][idx] if x!=PAD_INDEX]
            data["hyp"][idx] = [x for x in data["hyp"][idx] if x!=PAD_INDEX] 
            
            text.append(data["text"][idx])
            hyp.append(data["hyp"][idx])
            label.append(data["label"][idx])

        batches.append(nli_batch_to_variable(text, hyp, label))

    return batches

def random_batches(batch_size, pairs):
    n_batches = int(len(pairs) / batch_size)
    p = np.random.permutation(len(pairs))

    batches = []
    for i in range(n_batches):
        input_seqs = []
        target_seqs = []
        # Choose random pairs
        for j in range(batch_size):
            pair = pairs[p[i*batch_size+j]]
            input_seqs.append(pair[0])
            target_seqs.append(pair[1])

        batches.append(batch_pairs(input_seqs, target_seqs))

    return batches


def parse_query(msg, wd):
    return tokenize(separate(normalize(msg)), wd)


def clean_resp(raw_resp, rmv_tokens=list(RESERVED_W2I.keys())):
    if '<unk>' in rmv_tokens.keys():
        del rmv_tokens['<unk>']
    resp = [w for w in raw_resp if not w in rmv_tokens]
    return " ".join(resp)


def remove_punctuation(sequence):
    seq_out = []
    for s in sequence:
        if not s in PUNCTUATION:
            seq_out.append(s)
    return seq_out

def clean_seq(seq, remove_punctuation=False):
    seq_out = []
    for s in seq:
        if s not in RMV_TOKENS and (not remove_punctuation or s not in PUNCTUATION):
            seq_out.append(s)
    return seq_out



def import_csv(datafile, max_lines=-1, unk_thresh=5):
    lines = []
    wd = WordDict()
    print("Reading input...")
    with open(datafile, 'r') as infile:
        count = 0
        for line in infile:
            if max_lines > 0 and count >= max_lines:
                break
            split_line = separate(normalize(line))
            wd.add_sentence(split_line)
            lines.append(split_line)
            count += 1
    print("Input read.")
    print(str(len(lines)), "total lines.")
    print(str(wd.n_words), "total unique words.")

    unks = wd.remove_unknowns(unk_thresh)

    print(str(len(unks)), "words removed.", str(wd.n_words), "words remaining in vocabulary.")

    return lines, wd

def list_to_string(msg):
    return ",".join(msg)

def export_pairs(pairs, path):
    outfile = open(path, 'w')
    for pair in pairs:
        outfile.write("--\n"+list_to_string(pair[0])+"\n"+list_to_string(pair[1])+"\n")
    outfile.close()

def import_pairs(path):
    infile = open(path, 'r')
    pairs=[]
    done=False
    while not done:
        delim = infile.readline()
        if delim != "":
            msg = separate(normalize(infile.readline()))
            resp = separate(normalize(infile.readline()))
            pairs.append((msg, resp))
        else:
            done=True
    return pairs

def get_lines(datafile, max_lines=-1):
    lines = []
    print("Reading input...")
    with open(datafile, 'r') as infile:
        count = 0
        for line in infile:
            if max_lines > 0 and count >= max_lines:
                break
            split_line = separate(normalize(line))
            lines.append(split_line)
            count += 1
    print("Input read.")
    print(str(len(lines)), "total lines.")

    return lines


def nli_batches_without_label(batch_size, data):
    n_batches = int(data["length"] / batch_size)
    remaining = data["length"] % batch_size

    batches = []
    for i in range(n_batches):
        # Choose random pairs
        text = []
        hyp = []
        for j in range(batch_size):
            idx = i*batch_size+j
            text.append(data["text"][idx])
            hyp.append(data["hyp"][idx])

        batches.append(nli_batch_to_variable_without_label(text, hyp))

    text = []
    hyp = []    
    for k in range(remaining):
        idx = n_batches*batch_size + k
        text.append(data["text"][idx])
        hyp.append(data["hyp"][idx])
    if remaining>0:
        batches.append(nli_batch_to_variable_without_label(text, hyp))
    return batches

def nli_batch_to_variable_without_label(text, hyp, sort=True):
    if sort:
        # Zip into pairs, sort by length (descending), unzip
        batch = sorted(zip(text, hyp), key=lambda p: len(p[0]), reverse=True)
        text, hyp = zip(*batch)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    text_lengths = [len(s) for s in text]
    text_padded = [pad_seq(s, max(text_lengths)) for s in text]

    hyp_lengths = [len(s) for s in hyp]
    hyp_padded = [pad_seq(s, max(hyp_lengths)) for s in hyp]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    text = np.array(text_padded, dtype='long').transpose(1,0)
    hyp = np.array(hyp_padded, dtype='long').transpose(1,0)

    return (text, text_lengths, hyp, hyp_lengths)

