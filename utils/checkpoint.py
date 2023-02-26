import numpy as np
import torch

import pickle
import os

import utils.directory
import utils.checkpoint

class Checkpoint(object):
    """
    Early stopping to stop the training when the accuracy (or other metric) does not improve 
    after certain epochs.

    Parameters
    ----------
        patience (int): how many epochs to wait before stopping when loss is not improving.

        min_delta (int): minimum difference between new loss and old loss for new loss to be 
            considered as an improvement.

        path (str): Path for the checkpoint to be saved to.

        metric_eval (str): Name of the metric used.
    """

    def __init__(self, CONFIG, min_delta=0, verbose=True):

        # Type training
        self.type_train = CONFIG.type_train

        # Network 
        self.model_name  = CONFIG.name_model
        self.rnn_type    = CONFIG.rnn_type
        self.input_size  = CONFIG.input_size
        self.hidden_size = CONFIG.hidden_size
        self.num_layers  = CONFIG.num_layers
        self.num_classes = CONFIG.num_classes
        self.dropout     = CONFIG.dropout
        self.activation  = CONFIG.activation
        self.batch_norm  = CONFIG.batch_norm

        # Early stopping
        self.patience    = CONFIG.patience
        self.metric_eval = CONFIG.metric_eval

        self.save_file   = CONFIG.save_file + CONFIG.experiment + CONFIG.name_model

        # Me sirven?
        # R: no lo sé, tu dime.
        self.min_delta = min_delta
        self.verbose   = verbose

        self.counter    = 0 # Deberia guardarlo para saber en cual early stoppinh va
        self.min_loss   = None
        self.best_acc   = None
        self.early_stop = False

        self.val_acc_min = np.Inf

        self.train_loss          = []
        self.val_loss            = []
        self.train_acc           = [] 
        self.val_acc             = []
        self.checkpoint_early    = None
        self.checkpoint_training = None

    def save_init(self, loss_list, optimizer, scheduler, lr):       
        '''Guarda los parametros utilizados'''
        ## Files where the results will be saved 
        utils.directory.create_dir(self.save_file)
        
        dictionary_model = {'type_train' : self.type_train,
                            'rnn_type'   : self.rnn_type,
                            'input_size' : self.input_size,
                            'hidden_size': self.hidden_size,
                            'num_layers' : self.num_layers,
                            'num_classes': self.num_classes,
                            'dropout'    : self.dropout,
                            'activation' : self.activation,
                            'batch_norm' : self.batch_norm,
                            'loss_list'  : loss_list,
                            'optimizer'  : optimizer,
                            'scheduler'  : scheduler,
                            'lr'         : lr,
                            'patience'   : self.patience,
                            'metric_eval': self.metric_eval,
                            'counter'    : self.counter
                            }

        with open(self.save_file + '/parameters.model', 'wb') as f:
            pickle.dump(dictionary_model, f)

    def save_training(self, epoch, model, optimizer, lr_scheduler):
        self.type = 'training' 
        self.save_checkpoint(epoch, model, optimizer, lr_scheduler)     


    def early_stopping(self, epoch, model, optimizer, lr_scheduler):
        """
        Train_loss, train_acc, 
        val_loss y val_acc: Listas que contienen el valor promedio de cada epoca
        """
        self.type = 'early_stopping'

        if self.min_loss is None:

            self.min_loss = self.val_loss[-1]
            self.save_checkpoint(epoch, model, optimizer, lr_scheduler)
            
        
        elif self.val_loss[-1] < self.min_loss + self.min_delta:
        
            self.min_loss = self.val_loss[-1]
            self.counter  = 0 
            self.save_checkpoint(epoch, model, optimizer, lr_scheduler)
            
        
        elif self.val_loss[-1] > self.min_loss + self.min_delta:

            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} de {self.patience}")
            
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


    def save_checkpoint(self, epoch, model, optimizer, lr_scheduler):
        '''Saves model when metric evaluated decrease.'''

        checkpoint = {'epoch'     : epoch,
                      'train_loss': self.train_loss,
                      'val_loss'  : self.val_loss,
                      'train_acc' : self.train_acc,
                      'val_acc'   : self.val_acc,
                      'counter'   : self.counter,
                      }

        for i in range(len(model)):
            checkpoint[f'model_state_{i}'] = model[i].state_dict()
            checkpoint[f'optim_state_{i}'] = optimizer[i].state_dict()

        if self.type == 'early_stopping':
            if self.verbose:
                print(f'Saving model... {self.metric_eval} improved from {self.val_acc_min:.5f} to {self.val_acc[-1]:.5f}')

            self.create_checkpoint_file(self.checkpoint_early)
            self.checkpoint_early = self.save_file + f'/best_result/{self.model_name}pt-{epoch:03d}-{self.val_acc[-1]:.4f}.pt'
            checkpoint['checkpoint_early'] = checkpoint_path = self.checkpoint_early
            self.val_acc_min = self.val_acc[-1] 

        elif self.type == 'training':
            self.create_checkpoint_file(self.checkpoint_training)
            self.checkpoint_training = self.save_file + f'/save_training/{self.model_name}pt-{epoch:03d}.pt'
            checkpoint_path = self.checkpoint_training 

        torch.save(checkpoint, checkpoint_path)        

    def create_checkpoint_file(self, checkpoint_path):
        if checkpoint_path is not None:
            utils.directory.remove_dir(checkpoint_path)
        else:
            if self.type   == 'early_stopping': path_aux = self.save_file + '/best_result'
            elif self.type == 'training': path_aux = self.save_file + '/save_training'
            utils.directory.create_dir(path_aux)
