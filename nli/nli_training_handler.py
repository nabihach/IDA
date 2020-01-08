from nli_model import *
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt
import csv
import pdb

class TrainingHandler(object):
    def __init__(self, model, train_set, val_set_dict, learning_rate, save_dir, patience=10, annealing_factor=0.1,
                 weight_decay=0.01,clip=5.0, tf_ratio=1.0, initial_epoch=0):

        self.model=model
        self.train_set = train_set
        self.val_set_dict = val_set_dict

        self.clip = clip
        self.tf_ratio = tf_ratio
        self.save_dir = save_dir

        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.lr = learning_rate

        self.optim = []
        self.sched = []
        for parameter in self.model.parameter_list:
            
            # Initialize optimizers
            filtered_parameter = filter(lambda p: p.requires_grad, parameter)
            optimizer = optim.Adam(filtered_parameter, lr=learning_rate)

            # Add schedulers to reduce the learning rate if the loss plateaus
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=annealing_factor,
                                                           patience=patience)

            self.optim.append(optimizer)
            self.sched.append(scheduler)

        self.losses = []

    def train_model(self, epochs, batch_size, print_interval=1, save_interval=-1, tl_mode=False):
        eca = 0
        dca = 0

        loss_total = 0

        print("Beginning training...")
        if tl_mode is True:
            print("Transfer Learning mode is ON...")
        start = time.time()

        target_epoch = self.initial_epoch + epochs

        while self.epoch < target_epoch:
            self.epoch += 1

            # Get training data for this cycle
            batches = get_nli_batch(batch_size, self.train_set)

            n_batches = len(batches)

            for i in tqdm(range(n_batches)):
                loss, gradients = self._train_iter(batches[i], tl_mode=tl_mode)

                loss_total += loss

            loss_avg = loss_total / n_batches
            loss_total = 0

            loss_dict = { "train": loss_avg }

            # Calculate validation loss
            if self.val_set_dict is not None:
                val_loss_avg = self._get_val_loss()
                loss_dict.update(val_loss_avg)

            self.losses.append(loss_dict)

            # Print progress and loss every X epochs
            if print_interval > 0:
                if self.epoch % print_interval == 0:
                    print_summary = '-' * 40 + '\nEPOCH #%d SUMMARY:\nTotal time spent (time left): %s, Loss: %s'\
                                    % (self.epoch, time_since(start, (self.epoch-self.initial_epoch) / epochs), loss_dict)
                    self._print_log(print_summary)

            # Calculate BLEU score and save checkpoint every Y epochs
            if self.epoch < target_epoch:
                if save_interval > 0:
                    if self.epoch % save_interval == 0:
                        name = "_" + str(self.epoch) + ".tar"
                        self._save_checkpoint(self.save_dir, name)
            else:
                name = "_" + str(self.epoch) + ".tar"
                self._save_checkpoint(self.save_dir, name, save_loss=False)

    def _train_iter(self, batch, tl_mode=False):
        # Zero the gradients at the start of the training pass
        
        for optim in self.optim:
            optim.zero_grad()

        # Run the train function
        loss = self.model.train_batch(batch)

        # Clip gradient norms
        gradients = []
        for parameters in self.model.parameter_list:
           clipped_gradient = torch.nn.utils.clip_grad_norm_(parameters, self.clip)
           gradients.append(clipped_gradient)

        # Update parameters with optimizers
        for optim in self.optim:
            optim.step()

        return loss, gradients

    def _get_loss(self, dataset):
        loss_total = 0.0
        batch_size = 32
        batches = nli_random_batches(batch_size, dataset)
        n_batches = len(batches)
        for i in range(n_batches):
            loss = self.model.validate(batches[i])
            loss_total += loss
        loss_avg = loss_total / n_batches
        return loss_avg

    def _get_val_loss(self):
        val_loss_dict = {}
        for val_name in self.val_set_dict:
            val_set = self.val_set_dict[val_name]
            val_loss_dict[val_name] = self._get_loss(val_set)

        return val_loss_dict

    def _print_log(self, print_summary):
        save_logs(print_summary, self.save_dir)
        print(print_summary)
        for i, p in enumerate(self.optim[1].param_groups):
            save_logs(p['lr'], self.save_dir)
            print('LR: ', p['lr'])

    def _save_checkpoint(self, save_dir, name, save_loss=False):
        accuracy = {}

        for val_name in self.val_set_dict:
            val_set = self.val_set_dict[val_name]
            accuracy[val_name] = self.model.score(val_set)

        print("Accuracy:", accuracy)
        save_scores(accuracy, save_dir)

        if save_loss == True:
            fig_out = save_dir + FIG_FILE
            df_out = save_dir + LOSS_FILE

            self._save_losses(df_out)
            if not USE_CUDA:
                (self._plot_losses(df_out)).savefig(fig_out)

        # Save model checkpoint
        self.model.export_state(save_dir, name, epoch=self.epoch)

    def _plot_losses(self, path):
        train_losses = []
        val_losses = []

        #read the loss file
        with open(path, 'rt') as loss_file:
            loss_rows = csv.reader(loss_file, delimiter=',')
            for row in loss_rows:
                train_losses.append(float(row[0]))
                val_losses.append(float(row[1]))

        fig = plt.figure()
        plt.plot(train_losses, color='red', label='Train_loss', marker='o')
        plt.plot(val_losses, color='blue', label='Val_loss', marker='o')
        plt.legend(loc='upper right', frameon=False)
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        return fig

    def _save_losses(self, path):
        outfile = open(path, 'a')
        for i in range(len(self.train_losses)):
            outfile.write(str(self.train_losses[i])+','+str(self.val_losses[i])+"\n")
        outfile.close()
