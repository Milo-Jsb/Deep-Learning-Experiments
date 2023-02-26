import torch

# Files.py
import utils.time_distribuited

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FeatureExtractor(torch.nn.Module):
    
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
