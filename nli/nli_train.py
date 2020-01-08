from nli_model import *
from nli_training_handler import *
from utils import *
import nli_preprocessor
import argparse
import os
import os.path
import torch
import pdb
import itertools



def init_parser():
    parser = argparse.ArgumentParser(description='NLI biLSTM model with progressive memory.')
    parser.add_argument('-max', dest='max_lines', action='store', type=int, default=-1)
    parser.add_argument('-e', dest='epochs', action='store', type=int, default=10)
    parser.add_argument('-hs', dest='hidden_size', action='store', type=int, default=300)
    parser.add_argument('-bs', dest='batch_size', action='store', type=int, default=32)
    parser.add_argument('-lr', dest='learning_rate', action='store', type=float, default=0.0003)
    parser.add_argument('-l', dest='layers', action='store',type=int, default=1)
    parser.add_argument('-u', dest='unk_thresh', action='store', type=int, default=0)
    parser.add_argument('-s', dest='save_interval', action='store', type=int, default=1)
    parser.add_argument('-p', dest='patience', action='store', type=int, default=100)
    parser.add_argument('-f', dest='annealing_factor', action='store',type=float, default=0.5)
    parser.add_argument('-wd', dest='weight_decay', action='store',type=float, default=0)
    parser.add_argument('-m', dest='model_type', action='store', default='')
    parser.add_argument('-t', dest='load_pretrained_model', action='store', type=str, default=None)
    parser.add_argument('-emb', dest='pretrained_embeddings', action='store',type=bool, default=True)
    parser.add_argument('-g', dest='genre', action='store', type=str, default=None)
    parser.add_argument('-tl', dest='transfer_learning_mode', action='store',type=bool, default=False)
    parser.add_argument('-tg', dest='target_genre', action='store', type=str, default=None)
    args = parser.parse_args()

    return args

def is_valid_pretrained_model(model_name):
    if os.path.exists(MODEL_DIR+model_name):
        return True
    return False

def create_model(args):
    # Fetch data
    train_set = nli_preprocessor.get_multinli_text_hyp_labels(max_lines=args.max_lines, genre=args.genre)
    # To get full training set of 392702 lines, use: train_set = nli_preprocessor.get_multinli_training_set(max_lines=args.max_lines)
    matched_val_set = nli_preprocessor.genre_val_set('fiction')
    
    #To get full matched validation set use: 
    #matched_val_set = nli_preprocessor.get_multinli_matched_val_set()
    
    mismatched_val_set = nli_preprocessor.genre_val_set('government')
    #To get full mismatched validation set use: 
    #mismatched_val_set = nli_preprocessor.get_multinli_mismatched_val_set()

    model_path = init_save(args, (train_set, matched_val_set, mismatched_val_set))

    # Create WordDict from all data
    unmerged_sentences = []
    for data in [train_set, matched_val_set, mismatched_val_set]:
        unmerged_sentences.extend( [ data["text"], data["hyp"]  ] )
    all_sentences = list(itertools.chain.from_iterable(unmerged_sentences))
    wd = nli_preprocessor.get_word_dict(all_sentences, args.max_lines, args.unk_thresh)

    # Tokenize sentences
    for data in [train_set, matched_val_set, mismatched_val_set]:
        data["text"] = nli_preprocessor.tokenize_sentences(data["text"], wd)
        data["hyp"] = nli_preprocessor.tokenize_sentences(data["hyp"], wd)

    print("Variables processed.")

    # Initialize the model
    if args.model_type == 'GRU' or args.model_type == '':
        base_rnn = torch.nn.GRU
    elif args.model_type == 'LSTM':
        base_rnn = torch.nn.LSTM
    else:
        raise NotImplementedError()

    model = BiLSTMModel().init_model(wd, args.hidden_size, args.layers, args.layers, base_rnn, args.pretrained_embeddings)

    val_set_dict = {"matched_val": matched_val_set, "mismatched_val": mismatched_val_set}

    return model, model_path, train_set, val_set_dict

def load_model(args):
    model_path = MODEL_DIR + args.load_pretrained_model
    model, initial_epoch = BiLSTMModel().import_state(model_path, load_epoch=True)
    model_dir=os.path.dirname(model_path)+"/"

    if args.transfer_learning_mode==True:
        print("Adding new vocabulary...")
        model.add_new_vocabulary(args.target_genre)
        print("Resizing memory pad ... ")
        model.encoder.add_target_pad()
        #model.encoder.add_target_pad_2() ---> Use this instead of line 94 if you want to add a 2nd target pad
        #model.encoder.add_target_pad_3() ---> Use this instead of line 94 if you want to add a 3rd target pad
        #model.encoder.add_target_pad_4() ---> Use this instead of line 94 if you want to add a 4th target pad       
        train_set = nli_preprocessor.get_multinli_text_hyp_labels(max_lines=args.max_lines, genre=args.target_genre)
        # To use full matched validation set: use matched_val_set = nli_preprocessor.get_multinli_matched_val_set()
        matched_val_set = nli_preprocessor.genre_val_set('fiction')
        #To use full mismatched validation set use: mismatched_val_set = nli_preprocessor.get_multinli_mismatched_val_set()
        mismatched_val_set = nli_preprocessor.genre_val_set('government')
        for data in [train_set, matched_val_set, mismatched_val_set]:
            data["text"] = nli_preprocessor.tokenize_sentences(data["text"], model.wd)
            data["hyp"] = nli_preprocessor.tokenize_sentences(data["hyp"], model.wd)
        save_new_train_val_sets((train_set, matched_val_set, mismatched_val_set), model_dir)

    else:
        train_set = import_data(model_dir + "/" + TRAIN_FILE)
        matched_val_set = import_data(model_dir + "/" + VAL_1_FILE)
        mismatched_val_set = import_data(model_dir + "/" + VAL_2_FILE)

        # Tokenize sentences
        for data in [train_set, matched_val_set, mismatched_val_set]:
            data["text"] = nli_preprocessor.tokenize_sentences(data["text"], model.wd)
            data["hyp"] = nli_preprocessor.tokenize_sentences(data["hyp"], model.wd)

    init_save_pretrained_model(args, model_dir)
    val_set_dict = {"matched_val": matched_val_set, "mismatched_val": mismatched_val_set}

    return model, model_dir, initial_epoch, train_set, val_set_dict



if __name__ == '__main__':
    args = init_parser()

    if args.load_pretrained_model is not None:
        if is_valid_pretrained_model(args.load_pretrained_model):
            model, model_dir, initial_epoch, train_set, val_set_dict = load_model(args)

            print("Loaded previous model: "+model_dir)
        else:
            print("The model " + str(args.load_pretrained_model) + " does not exist. Please load a valid model.")
    else:
        model, model_dir, train_set, val_set_dict = create_model(args)
        initial_epoch = 0

    trainer = TrainingHandler(model, train_set, val_set_dict, args.learning_rate, model_dir,
                              patience=args.patience, annealing_factor=args.annealing_factor,
                              weight_decay=args.weight_decay, initial_epoch=initial_epoch)

    # Train the model
    trainer.train_model(args.epochs, args.batch_size, save_interval=args.save_interval, tl_mode=args.transfer_learning_mode)
