import numpy as np
import torch

# Files.py
import utils.time_distribuited
import models.mme.grad_reverse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

##############################################################################################################
class RAPIDModel(torch.nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, num_classes,
                dropout, activation, batch_norm):
        super(RAPIDModel, self).__init__()

        # Paremeters
        self.input_size   = input_size   #
        self.hidden_size  = hidden_size  #16
        self.num_layers   = num_layers   #2
        self.num_classes  = num_classes  #2

        # First Layer
        self.gru_1 = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.bn_1 = torch.nn.BatchNorm1d(50)
        
        # Second Layer
        self.gru_2 = torch.nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.bn_2 = torch.nn.BatchNorm1d(50)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

        # Dense
        self.fc      = torch.nn.Linear(self.hidden_size, self.num_classes)

        # Initialize as Keras
        self.init_weights()

    def forward(self, x):
        x, h_n = self.gru_1(x)
        x = self.dropout(x)
        x = self.bn_1(x)

        x, h_n = self.gru_2(x)
        x = self.dropout(x)
        x = self.bn_2(x)
        
        x = self.dropout(x)
        
        out = utils.time_distribuited.TimeDistributedRAPID(self.fc, batch_first=True)(x)

        return out

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights.
        https://gist.github.com/thomwolf/eea8989cab5ac49919df95f6f1309d80
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b  = (param.data for name, param in self.named_parameters() if 'bias' in name)
        
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)
    
    def __train_normal(self, train_dataloader, loss_list, acc_list):
        
        with torch.no_grad():
        
            for batch_idx, (data, targets) in enumerate(train_dataloader):
                    
                # Get data in cuda if it was possible
                X_train_sorted = data.float().to(device)

                y_train_sorted = targets.to(device)

                # forward
                scores_sorted  = self.model[0](X_train_sorted)

                loss  = self.loss[0](scores_sorted, y_train_sorted)          

                # backward
                self.optimizer[0].zero_grad() # Quizas tenga que poner [0]
                loss.backward()

                # gradient descent or adam step
                self.optimizer[0].step()

                # metrics 
                acc = self.__get_metrics(y_train_sorted, scores_sorted)

                # save metrics for step
                loss_list.append(loss.item()) # standard Python number
                acc_list.append(acc)
            
            return loss_list, acc_list
    
    def __validation_normal(self, val_dataloader, loss_list, acc_list):
        
        with torch.no_grad():
                
            for batch_idx, (data, targets) in enumerate(val_dataloader):
                # Get data to cuda if possible
                X_val_sorted = data.float().to(device)
                y_val_sorted = targets.to(device)

                # forward
                scores_sorted = self.model[0](X_val_sorted)
                loss          = self.loss[0](scores_sorted, y_val_sorted)
            
                # metrics 
                acc = self.__get_metrics_rapid(y_val_sorted, scores_sorted)

                # save metrics for step
                loss_list.append(loss.item()) # standard Python number
                acc_list.append(acc)

            return loss_list, acc_list
        
    def __get_metrics(self, y_true, scores):
        
        y_pred = torch.argmax(scores, axis=1)

        # Try if y_true is list or one-hot 
        try:
            acc = (y_true == y_pred).sum().item() /  np.prod(y_true.shape) 
        except RuntimeError:
            y_true = torch.argmax(y_true, dim=1)
            acc = (y_true == y_pred).sum().item() / np.prod(y_true.shape)        

        if self.metric_eval.lower() == 'f1_score':
            
            f1_score = f1_score(y_true=y_true.tolist(), y_pred=y_pred.tolist(), 
                                average='macro', pos_label=None)   

            return acc, f1_score     

        return acc
##############################################################################################################
class FeatureExtractor(torch.nn.Module):
    """ 
    Feature Extractor for the MME RAPID model. 
    """
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        super(FeatureExtractor, self).__init__()

        # Paremetros
        self.input_size   = input_size   #
        self.hidden_size  = hidden_size  #16
        self.num_layers   = num_layers   #2

        # First Layer
        self.gru_1 = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.bn_1  = torch.nn.BatchNorm1d(50)
        
        # Second Layer
        self.gru_2 = torch.nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.bn_2  = torch.nn.BatchNorm1d(50)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

        # Inicializa pesos como en keras
        self.init_weights()

    def forward(self, x):
        # First layer
        x, h_n = self.gru_1(x) #(h_0, c_0))
        x = self.dropout(x)
        x = self.bn_1(x)
        # Second layer
        x, h_n = self.gru_2(x) #(h_0, c_0))
        x = self.dropout(x)
        x = self.bn_2(x)
        # Last dropout
        x = self.dropout(x)

        return x

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights.
        https://gist.github.com/thomwolf/eea8989cab5ac49919df95f6f1309d80
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b  = (param.data for name, param in self.named_parameters() if 'bias' in name)
        #torch.nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

class Classifier(torch.nn.Module):
    """ 
    Classifier network for the MME RAPID model. 
    """
    def __init__(self, hidden_size, num_classes, temp):
        
        super(Classifier, self).__init__()
        
        # Parametros faltantes
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Capa densa del modelo
        self.fc          = torch.nn.Linear(self.hidden_size, self.num_classes)
        
        # Parametro de temperatura
        self.temp        = temp

    
    def forward(self, x, grad_rev=True, eta=1.0):
        # Se implementa el gradiente reverso
        if grad_rev:
            x = models.mme.grad_reverse.grad_reverse(x, eta)        
        
        y     = utils.time_distribuited.TimeDistributedRAPID(self.fc, batch_first=True)(x)
        y_out = y / self.temp

        return y_out
    
##############################################################################################################