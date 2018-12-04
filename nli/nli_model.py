import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import nli_preprocessor
from utils import *
from preprocess import *
from encoder import EncoderRNN
import torch.nn as nn
from torch.autograd import Variable
import pickle
import tarfile
from tqdm import tqdm
import itertools



def import_data(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def export_data(data, path):
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()

def init_save(args, data, model_dir=MODEL_DIR):
    t = datetime.datetime.now()
    timestamp = str(t.day) + "_" + str(t.hour) + "_" + str(t.minute)
    path = model_dir + "mnli_" + timestamp + "/"
    if not os.path.isdir(path):
        os.mkdir(path)

    log_file = open(path + LOG_FILE, 'a')
    arg_dict = vars(args)
    log_file.write("Training Parameters:\n")
    for (arg, val) in arg_dict.items():
        log_file.write(str(arg) + ": " + str(val) + '\n')
    log_file.write("\n")
    log_file.close()
    print("Model parameters saved.")

    train, val_1, val_2 = data
    export_data(train, path + TRAIN_FILE)
    export_data(val_1, path+VAL_1_FILE)
    export_data(val_2, path+VAL_2_FILE)
    print("Training and validation sets saved.")

    return path

def save_new_train_val_sets(data, model_dir=MODEL_DIR):
    train, val_1, val_2 = data
    export_data(train, model_dir + TRAIN_FILE)
    export_data(val_1, model_dir + VAL_1_FILE)
    export_data(val_2, model_dir + VAL_2_FILE)
    print("New training and validation sets saved.")


class BiLSTMModel(nn.Module):
    def __init__(self):
        super(BiLSTMModel,self).__init__()
        self.base_rnn = None
        self.wd = None

    def init_model(self, wd, hidden_size, e_layers, d_layers, base_rnn, pretrained_embeddings=None, dropout_p=0.1):

        self.base_rnn = base_rnn
        self.wd = wd
        self.dropout_p = dropout_p
        if pretrained_embeddings is True:
            print("Loading GloVe Embeddings ...")
            pretrained_embeddings = load_glove_embeddings(wd.word2index, hidden_size)

        self.encoder = EncoderRNN(wd.n_words, hidden_size, n_layers=e_layers, base_rnn=base_rnn, pretrained_embeddings=pretrained_embeddings)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(int(hidden_size * 8), int(hidden_size)),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.Linear(int(hidden_size), 3),
            torch.nn.Softmax(dim=1)
        )
        self.parameter_list = [self.encoder.parameters(), self.mlp.parameters()]

        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.mlp = self.mlp.cuda()

        return self

    def forward(self, batch, inference=False):
        # Convert batch from numpy to torch
        if inference is True:
            text_batch, text_lengths, hyp_batch, hyp_lengths = batch
        else:
            text_batch, text_lengths, hyp_batch, hyp_lengths, labels = batch
        batch_size = text_batch.size(1)

        # Pass the input batch through the encoder
        text_enc_fwd_outputs, text_enc_bkwd_outputs, text_encoder_hidden = self.encoder(text_batch, text_lengths)
        hyp_enc_fwd_outputs, hyp_enc_bkwd_outputs, hyp_encoder_hidden = self.encoder(hyp_batch, hyp_lengths)

        last_text_enc_fwd = text_enc_fwd_outputs[-1, :, :]
        last_text_enc_bkwd = text_enc_bkwd_outputs[0, :, :]
        last_text_enc = torch.cat( (last_text_enc_fwd, last_text_enc_bkwd) , dim=1)
        last_hyp_enc_fwd = hyp_enc_fwd_outputs[-1, :, :]
        last_hyp_enc_bkwd = hyp_enc_bkwd_outputs[0, :, :]
        last_hyp_enc = torch.cat( (last_hyp_enc_fwd, last_hyp_enc_bkwd) , dim=1)

        mult_feature, diff_feature = last_text_enc * last_hyp_enc, torch.abs( last_text_enc - last_hyp_enc )

        features = torch.cat( [last_text_enc, last_hyp_enc, mult_feature, diff_feature], dim=1)
        outputs = self.mlp(features) # B x 3
        return outputs

    def get_loss_for_batch(self, batch):
        labels = batch[-1]
        outputs = self(batch)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn( outputs, labels )

        return loss

    def torch_batch_from_numpy_batch(self, batch):
        batch = list(batch)

        variable_indices = [0, 2, 4] # tuple indices of variables need to be converted
        for i in variable_indices:
            var = Variable(torch.from_numpy(batch[i]))
            if USE_CUDA:
                var = var.cuda()
            batch[i] = var

        return batch

    # Trains on a single batch
    def train_batch(self, batch, tl_mode=False):
        self.train()

        batch = self.torch_batch_from_numpy_batch(batch)
        loss = self.get_loss_for_batch(batch)
        loss.backward()

        return loss.item() #loss.data[0]

    def validate(self, batch):
        self.eval()

        batch = self.torch_batch_from_numpy_batch(batch)
        return self.get_loss_for_batch(batch).item() #.data[0]

    def score(self, data):
        batch_size = 1
        batches = nli_batches(batch_size, data)

        total_correct = 0
        for batch in tqdm(batches):
            batch = self.torch_batch_from_numpy_batch(batch)
            num_correct = self._acc_for_batch(batch)
            total_correct += num_correct

        acc = total_correct / (len(batches) * batch_size)

        return acc

    def _acc_for_batch(self, batch):
        '''
        :param batch:
        :return: The number of correct predictions in a batch
        '''
        self.eval()

        outputs = self(batch)
        predictions = outputs.max(1)[1]

        labels = batch[-1]

        num_error = torch.nonzero(labels - predictions)
        num_correct = labels.size(0) - num_error.size(0)

        return num_correct

    def export_state(self, dir, label, epoch=-1):
        print("Saving models.")

        cwd = os.getcwd() + '/'

        enc_out = dir + ENC_1_FILE
        mlp_out = dir + MLP_FILE
        i2w_out = dir + I2W_FILE
        w2i_out = dir + W2I_FILE
        w2c_out = dir + W2C_FILE
        inf_out = dir + INF_FILE

        torch.save(self.encoder.state_dict(), enc_out)
        torch.save(self.mlp.state_dict(), mlp_out)

        i2w = open(i2w_out, 'wb')
        pickle.dump(self.wd.index2word, i2w)
        i2w.close()
        w2i = open(w2i_out, 'wb')
        pickle.dump(self.wd.word2index, w2i)
        w2i.close()
        w2c = open(w2c_out, 'wb')
        pickle.dump(self.wd.word2count, w2c)
        w2c.close()

        info = open(inf_out, 'w')
        using_lstm = 1 if self.base_rnn == nn.LSTM else 0
        info.write(str(self.encoder.hidden_size) + "\n" + str(self.encoder.n_layers) + "\n" + 
            str(self.wd.n_words) + "\n" + str(using_lstm))            
        if epoch > 0:
            info.write("\n"+str(epoch))
        info.close()

        files = [enc_out, mlp_out, i2w_out, w2i_out, w2c_out, inf_out]


        print("Bundling models")

        tf = tarfile.open(cwd + dir + label, mode='w')
        for file in files:
            tf.add(file)
        tf.close()

        for file in files:
            os.remove(file)

        print("Finished saving models.")


    def import_state(self, model_file, active_dir=TEMP_DIR, load_epoch=False):
        print("Loading models.")
        cwd = os.getcwd() + '/'
        tf = tarfile.open(model_file)

        # extract directly to current model directory
        for member in tf.getmembers():
            if member.isreg():
                member.name = os.path.basename(member.name)
                tf.extract(member, path=active_dir)

        info = open(active_dir + INF_FILE, 'r')
        lns = info.readlines()
        hidden_size, e_layers, n_words, using_lstm = [int(i) for i in lns[:4]]

        if load_epoch:
            epoch = int(lns[-1])

        i2w = open(cwd + TEMP_DIR + I2W_FILE, 'rb')
        w2i = open(cwd + TEMP_DIR + W2I_FILE, 'rb')
        w2c = open(cwd + TEMP_DIR + W2C_FILE, 'rb')
        i2w_dict = pickle.load(i2w)
        w2i_dict = pickle.load(w2i)
        w2c_dict = pickle.load(w2c)
        wd = WordDict(dicts=[w2i_dict, i2w_dict, w2c_dict, n_words])
        w2i.close()
        i2w.close()
        w2c.close()

        self.base_rnn = nn.LSTM if using_lstm == 1 else nn.GRU
        self.wd = wd
        self.encoder = EncoderRNN(wd.n_words, hidden_size, n_layers=e_layers, base_rnn=self.base_rnn)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(int(hidden_size * 8), int(hidden_size)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(int(hidden_size), 3),
            torch.nn.Softmax(dim=1)
        )
        if not USE_CUDA:
            self.encoder.load_state_dict(torch.load(cwd + TEMP_DIR + ENC_1_FILE, map_location=lambda storage, loc: storage))
            self.mlp.load_state_dict(torch.load(cwd + TEMP_DIR + MLP_FILE, map_location=lambda storage, loc: storage))
        else:
            self.encoder.load_state_dict(torch.load(cwd + TEMP_DIR + ENC_1_FILE))
            self.mlp.load_state_dict(torch.load(cwd + TEMP_DIR + MLP_FILE))
            self.encoder = self.encoder.cuda()
            self.mlp = self.mlp.cuda()

        self.encoder.eval()
        self.mlp.eval()

        self.parameter_list = [self.encoder.parameters(), self.mlp.parameters()]
        tf.close()

        print("Loaded models.")

        if load_epoch:
            return self, epoch
        else:
            return self


    def torch_batch_from_numpy_batch_without_label(self, batch):
        batch = list(batch)

        variable_indices = [0, 2]
        for i in variable_indices:
            var = Variable(torch.from_numpy(batch[i]))
            if USE_CUDA:
                var = var.cuda()
            batch[i] = var

        return batch

    def predict(self, data):
        batch_size = 1
        batches = nli_batches_without_label(batch_size, data)

        predictions = []
        for batch in tqdm(batches):
            batch = self.torch_batch_from_numpy_batch_without_label(batch)
            outputs = self(batch, inference=True)
            pred = outputs.max(1)[1]
            predictions.append(pred)

        return torch.cat(predictions)

    def add_new_vocabulary(self, genre):
        old_vocab_size = self.wd.n_words
        print("Previous vocabulary size: "+str(old_vocab_size))

        train_set = nli_preprocessor.get_multinli_text_hyp_labels(genre=genre)#nli_preprocessor.get_multinli_training_set(max_lines=args.max_lines)
        matched_val_set = nli_preprocessor.get_multinli_matched_val_set()#genre_val_set(genre)

        unmerged_sentences = []
        for data in [train_set, matched_val_set]:
            unmerged_sentences.extend( [ data["text"], data["hyp"]  ] )
        all_sentences = list(itertools.chain.from_iterable(unmerged_sentences))

        for line in all_sentences:
            self.wd.add_sentence(line)

        print("New vocabulary size: "+str(self.wd.n_words))

        print("Extending the Embedding layer with new vocabulary...")
        num_new_words = self.wd.n_words - old_vocab_size
        self.encoder.extend_embedding_layer(self.wd.word2index, num_new_words)

        self.new_vocab_size = num_new_words

    def freeze_source_params(self):
        for name, param in self.named_parameters():
            if "rnn" in name:
                param.requires_grad = False
            if ("M_k" in name or "M_v" in name) and "target_4" not in name:
                param.requires_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad is True:
                print(name)
     
