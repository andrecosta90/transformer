import torch.nn as nn
import torch


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    '''
    What is Label Smoothing?
    
        Imagine you're teaching a child to identify animals, but instead of telling them that a picture is 
            definitely a "dog" (100% certain), you say it's a "dog" with 90% certainty and maybe a "cat" with 
            5% certainty, and a "fox" with 5% certainty. This helps the child not be too confident and be a 
            bit more flexible in their thinking.
        
        In machine learning, Label Smoothing does something similar. Instead of telling the model that the 
            correct answer is 100% correct (and all others are 0% correct), we spread out a little bit of 
            the "correctness" to other possible answers. This helps the model to not be overly confident 
            and generalize better.
    '''
   

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()

        # Fill all predictions with a small value.
        true_dist.fill_(self.smoothing / (self.size - 2))

        # Replace the correct label's probability with the confidence value (e.g., 90%).
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # Set the probability for the padding label to zero.
        true_dist[:, self.padding_idx] = 0

        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        '''
        By combining clone() and detach(), we create a new tensor that is both:
            + `.clone()` - Independent in memory: Changes to this tensor won't affect the original true_dist.
            + `.detach()` - Disconnected from the computational graph: This tensor won't affect gradient calculations during backpropagation.
        '''
        return self.criterion(x, true_dist.clone().detach())
