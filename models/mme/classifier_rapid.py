import torch
import models.mme.grad_reverse

import utils.time_distribuited

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    