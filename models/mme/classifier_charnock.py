import torch
import models.mme.grad_reverse

import utils.time_distribuited

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Classifier(torch.nn.Module):
    """ 
    Classifier network for the MME Charnock model. 
    """
    def __init__(self, hidden_size, num_classes, temp):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.fc = torch.nn.Linear(self.hidden_size, self.num_classes)
        
        self.temp = temp

    def forward(self, unpacked_out, unpacked_out_len, mask_1, grad_rev=False, eta=1.0):
        if grad_rev:
            unpacked_out = models.mme.grad_reverse.grad_reverse(unpacked_out, eta)

        unpacked_out = torch.nn.functional.normalize(unpacked_out * mask_1)

        out = utils.time_distribuited.TimeDistributedCharnock(self.fc, batch_first=True)(unpacked_out * mask_1)
        mask_2 = mask_1[::,::,:out.size(2)] 
        y = (out * mask_2).sum(1) / unpacked_out_len.unsqueeze(-1).float().to(device)

        y_out = y / self.temp
        
        return y_out