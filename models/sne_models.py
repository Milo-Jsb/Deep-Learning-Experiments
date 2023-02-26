import torch

# Files.py
import utils.time_distribuited

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CharnockModel(torch.nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, num_classes,
                dropout, activation, batch_norm):
        super(CharnockModel, self).__init__()

        # RNN type to use
        if rnn_type.lower() == 'rnn':
            RNNLayer = torch.nn.RNN
        
        elif rnn_type.lower() == 'lstm':
            RNNLayer = torch.nn.LSTM
        
        elif rnn_type.lower() == 'gru':
            RNNLayer = torch.nn.GRU

        # Paremeters
        self.input_size  = input_size    #14
        self.hidden_size = hidden_size  #16
        self.num_layers  = num_layers    #2
        self.num_classes = num_classes  #2
        self.dropout     = dropout
        
        # Network
        self.lstm           = RNNLayer(self.input_size, self.hidden_size, self.num_layers, 
                                      batch_first=True, dropout=self.dropout)
        
        self.output_dropout = torch.nn.Dropout(self.dropout)
        
        self.fc             = torch.nn.Linear(self.hidden_size, self.num_classes)

        # Init weights as keras
        self.init_weights()

    def forward(self, x, batch_size):
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

class RAPIDModel(torch.nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, num_classes,
                dropout, activation, batch_norm):
        super(RAPIDModel, self).__init__()

        # Paremetros
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

        # ?
        self.fc      = torch.nn.Linear(self.hidden_size, self.num_classes)

        # Inicializa pesos como en keras
        self.init_weights()

    def forward(self, x):
        x, h_n = self.gru_1(x) #(h_0, c_0))
        x = self.dropout(x)
        x = self.bn_1(x)

        x, h_n = self.gru_2(x) #(h_0, c_0))
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
        #torch.nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

