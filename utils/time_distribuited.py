import torch

class TimeDistributedCharnock(torch.nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributedCharnock, self).__init__()
        self.module      = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash batch and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (batch * timesteps, input_size). 
        
        # La fc tomara la dimension del batch como (batch * timesteps)
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (batch, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))               # (timesteps, batch, output_size)

        return y

class TimeDistributedRAPID(torch.nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributedRAPID, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash batch and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (batch * timesteps, input_size). 
        
        # La fc tomara la dimension del batch como (batch * timesteps)
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), y.size(-1), -1)  # (batch, output_size, timesteps)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, batch, output_size)

        return y