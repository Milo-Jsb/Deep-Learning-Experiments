import torch

# Files.py
import utils.time_distribuited

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CharnockModel(torch.nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, num_classes,
                dropout, activation, batch_norm):
        super(CharnockModel, self).__init__()

        # SE ESCOGE LA RNN A UTILIZAR 
        if rnn_type.lower() == 'rnn':
            RNNLayer = torch.nn.RNN
        elif rnn_type.lower() == 'lstm':
            RNNLayer = torch.nn.LSTM
        elif rnn_type.lower() == 'gru':
            RNNLayer = torch.nn.GRU

        # Paremetros
        self.input_size = input_size   #14
        self.hidden_size = hidden_size #16
        self.num_layers = num_layers   #2
        self.num_classes = num_classes #2

        self.dropout = dropout
        #self.activation = activation
        #self.batch_norm = batch_norm

        self.lstm = RNNLayer(self.input_size, self.hidden_size, self.num_layers, 
                            batch_first=True, dropout=self.dropout)
        self.output_dropout = torch.nn.Dropout(self.dropout)
        self.fc = torch.nn.Linear(self.hidden_size, self.num_classes)

        # Inicializa pesos como en keras
        self.init_weights()

    def forward(self, x, batch_size):
        #h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        #c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        packed_out, (h_n, c_n) = self.lstm(x) #(h_0, c_0)
        out_dropout = self.output_dropout(packed_out.data)
        packed_out_drop = torch.nn.utils.rnn.PackedSequence(out_dropout, packed_out.batch_sizes)
        
        unpacked_out, unpacked_out_len = torch.nn.utils.rnn.pad_packed_sequence(packed_out_drop, padding_value=-999.0, batch_first=True)
        mask_1 = (unpacked_out != -999.0).type(torch.ByteTensor).to(device)
        out = utils.time_distribuited.TimeDistributedCharnock(self.fc, batch_first=True)(unpacked_out * mask_1)
        
        mask_2 = mask_1[::,::,:out.size(2)] 
        y = (out * mask_2).sum(1) / unpacked_out_len.unsqueeze(-1).float().to(device)

        return y

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
    
    def __train_normal(self, train_dataloader, loss_list, acc_list ):
        
        with torch.no_grad():

            for batch_idx, (data, targets, idx_sorted) in enumerate(train_dataloader):
                        
                # Get data in cuda if it was possible
                X_train_sorted = data.to(device)
                y_train_sorted = torch.stack(targets).to(device)

                batch_size = train_dataloader.batch_size

                # forward
                scores_sorted  = self.model[0](X_train_sorted, batch_size)
                loss           = self.loss[0](scores_sorted, y_train_sorted)
                
                scores         = torch.zeros(size=scores_sorted.size())
                y_train        = torch.zeros(size=y_train_sorted.size()) 
                
                for i in range(len(idx_sorted)):
                    scores[idx_sorted[i]]  = scores_sorted[i]
                    y_train[idx_sorted[i]] = y_train_sorted[i]           

                # backward
                self.optimizer[0].zero_grad() # Quizas tenga que poner [0]
                loss.backward()

                # gradient descent or adam step
                self.optimizer[0].step()

                # metrics 
                acc = self.__get_metrics(y_train, scores)

                # save metrics for step
                loss_list.append(loss.item()) # standard Python number
                acc_list.append(acc)
            
            return loss_list, acc_list
    
    def __validation_normal(self, val_dataloader, loss_list, acc_list):

        with torch.no_grad():

            for batch_idx, (data, targets, idx_sorted) in enumerate(val_dataloader):
                # Get data to cuda if possible
                X_val_sorted = data.to(device)
                y_val_sorted = torch.stack(targets).to(device)

                batch_size = val_dataloader.batch_size

                # forward
                scores_sorted = self.model[0](X_val_sorted, batch_size)
                loss = self.loss[0](scores_sorted, y_val_sorted)
                
                # Vuelve las predicciones y etiquetas al orden normal
                # sirve para evaluar las metricas
                scores = torch.zeros(size=scores_sorted.size())
                y_val = torch.zeros(size=y_val_sorted.size()) 
                
                for i in range(len(idx_sorted)):
                    scores[idx_sorted[i]] = scores_sorted[i]
                    y_val[idx_sorted[i]]  = y_val_sorted[i]
            
                # metrics 
                acc = self.__get_metrics(y_val, scores)

                # save metrics for step
                loss_list.append(loss.item())
                acc_list.append(acc)

            return loss_list, acc_list
    
    def __get_metrics(self, y_true, scores):
        
        y_pred = torch.argmax(scores, axis=1)

        # Try if y_true is list or one-hot 
        try:
            acc = (y_true == y_pred).sum().item() / len(y_true)
        except RuntimeError:
            y_true = torch.argmax(y_true, dim=1)
            acc = (y_true == y_pred).sum().item() / len(y_true)       

        if self.metric_eval.lower() == 'f1_score':
            #f1_score_custom = utils.F1Score('macro')(y_val_pred, y_val)
            f1_score = f1_score(y_true=y_true.tolist(), y_pred=y_pred.tolist(), 
                                average='macro', pos_label=None)   

            return acc, f1_score     

        return acc