import pandas as pd
import os
import sys
import torch
import json

import preprocess
import pdb

def tokenize_sentences(sentences, wd):
    tokenize_sentence = lambda s : wd.to_indices(s) + [preprocess.EOS_INDEX]

    return [ tokenize_sentence(s) for s in sentences ]

def get_word_dict(sentences, max_lines=-1, unk_thresh=1):
    print("Generating WordDict")
    wd = preprocess.WordDict()

    for i, sentence in enumerate(sentences):
        if max_lines > 0 and i >= max_lines:
            break

        wd.add_sentence(sentence)

    print(str(wd.n_words), "total unique words.")

    unks = wd.remove_unknowns(unk_thresh)

    print(str(len(unks)), "words removed.", str(wd.n_words), "words remaining in vocabulary.")

    return wd

def valid_label(label):
    labels = ['entailment', 'neutral', 'contradiction']
    return label in labels

def label_to_vector(label):
    if label == 'entailment':
        return 0
    elif label == 'neutral':
        return 1
    elif label == 'contradiction':
        return 2

    raise Exception('Unknown label:' + str(label))

def genre_val_set(genre):
    genre_val_path = "nli/data/raw/multinli_0.9/val_"+str(genre)+".jsonl"
    return get_nli_text_hyp_labels(genre_val_path)

def genre_test_set(genre):
    genre_test_path = "nli/data/raw/multinli_0.9/test_"+str(genre)+".jsonl"
    return get_nli_text_hyp_labels(genre_test_path)

def get_multinli_test_set_matched(max_lines=-1):
    multinli_test_path = "nli/data/raw/multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl"
    return get_nli_text_hyp(multinli_test_path, max_lines=max_lines)

def get_multinli_test_set_mismatched(max_lines=-1):
    multinli_test_path = "nli/data/raw/multinli_0.9/multinli_0.9_test_mismatched_unlabeled.jsonl"
    return get_nli_text_hyp(multinli_test_path, max_lines=max_lines)


def get_multinli_training_set(max_lines=-1):
    multinli_train_path = "nli/data/raw/multinli_0.9/multinli_0.9_train.jsonl"
    return get_nli_text_hyp_labels(multinli_train_path, max_lines=max_lines)

def get_multinli_matched_val_set():
    multinli_dev_matched_path = "nli/data/raw/multinli_0.9/multinli_0.9_dev_matched.jsonl"
    return get_nli_text_hyp_labels(multinli_dev_matched_path)

def get_multinli_mismatched_val_set():
    multinli_dev_mismatched_path = "nli/data/raw/multinli_0.9/multinli_0.9_dev_mismatched.jsonl"
    return get_nli_text_hyp_labels(multinli_dev_mismatched_path)

def dev_matched_text_hyp_labels(max_lines=-1, num_genres=-1, genre=None):
    multinli_path = "nli/data/raw/multinli_0.9/multinli_0.9_dev_matched.jsonl"

    data_filter = None
    
    all_genres = ['fiction', 'government', 'slate', 'telephone', 'travel', '9/11', 'face-to-face', 'letters', 'oup', 'verbatim']
    if genre is not None:
        sel_genres = [genre]
    else:
        sel_genres = all_genres
    data_filter = lambda data: data['genre'].map(lambda genre: genre in sel_genres)

    return get_nli_text_hyp_labels(multinli_path, max_lines=max_lines, data_filter=data_filter)

def dev_mismatched_text_hyp_labels(max_lines=-1, num_genres=-1, genre=None):
    multinli_path = "nli/data/raw/multinli_0.9/multinli_0.9_dev_mismatched.jsonl"

    data_filter = None
    
    all_genres = ['fiction', 'government', 'slate', 'telephone', 'travel', '9/11', 'face-to-face', 'letters', 'oup', 'verbatim']
    if genre is not None:
        sel_genres = [genre] 
    else:
        sel_genres = all_genres
    data_filter = lambda data: data['genre'].map(lambda genre: genre in sel_genres)

    return get_nli_text_hyp_labels(multinli_path, max_lines=max_lines, data_filter=data_filter)



def get_multinli_text_hyp_labels(max_lines=-1, num_genres=-1, genre=None):
    multinli_path = "nli/data/raw/multinli_0.9/multinli_0.9_train.jsonl"

    data_filter = None
    #if num_genres > 0:
    all_genres = ['fiction', 'government', 'slate', 'telephone', 'travel']
    if genre is not None:
        sel_genres = [genre] #['travel'] #all_genres[:num_genres]
    else:
        sel_genres = all_genres
    data_filter = lambda data: data['genre'].map(lambda genre: genre in sel_genres)

    return get_nli_text_hyp_labels(multinli_path, max_lines=max_lines, data_filter=data_filter)


def get_nli_text_hyp(path_to_nli_dataset, max_lines=-1, data_filter=None):

    nli_df = pd.read_json(path_to_nli_dataset, lines=True)
    
    valid_df = nli_df[ nli_df['sentence2'].notnull() ]
    if data_filter is not None:
        valid_df = valid_df[ data_filter(valid_df) ]

    reformat_sentence = lambda original_sentence: preprocess.separate(preprocess.normalize( original_sentence ))

    text_sentences = valid_df['sentence1'].map(reformat_sentence)
    hyp_sentences = valid_df['sentence2'].map(reformat_sentence)

    if 0 < max_lines and max_lines <= len(text_sentences):
        text_sentences, hyp_sentences = text_sentences[:max_lines], hyp_sentences[:max_lines]

    length = len(text_sentences)
    #print(length)
    return {"text": text_sentences, "hyp": hyp_sentences, "length": length}


def get_nli_text_hyp_labels(path_to_nli_dataset, max_lines=-1, data_filter=None):

    nli_df = pd.read_json(path_to_nli_dataset, lines=True)
    
    valid_df = nli_df[ nli_df['sentence2'].notnull() & nli_df['gold_label'].map(valid_label) ]
    if data_filter is not None:
        valid_df = valid_df[ data_filter(valid_df) ]

    reformat_sentence = lambda original_sentence: preprocess.separate(preprocess.normalize( original_sentence ))

    text_sentences = valid_df['sentence1'].map(reformat_sentence)
    hyp_sentences = valid_df['sentence2'].map(reformat_sentence)
    labels = valid_df['gold_label'].map( label_to_vector ).tolist()

    if 0 < max_lines and max_lines <= len(text_sentences):
        text_sentences, hyp_sentences, labels = text_sentences[:max_lines], hyp_sentences[:max_lines], labels[:max_lines]

    length = len(text_sentences)
    print(length)
    return {"text": text_sentences, "hyp": hyp_sentences, "label": labels, "length": length}


