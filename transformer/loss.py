from transformer.generator import Generator
from transformer.smoothing import LabelSmoothing


class SimpleLossCompute:
    "A simple loss compute and train function."

    '''
        x.contiguous().view(-1, x.size(-1))
            + x.contiguous(): Makes sure that the data in x is stored in a continuous block of memory. 
                This is important because some operations, like view(), work more efficiently (or only work) with contiguous memory.

        x.view(-1, x.size(-1)):
            + view(-1, x.size(-1)) reshapes the tensor x.
            + The -1 in the view() function means "infer the size for this dimension based on the other dimensions."
            + x.size(-1) is the size of the last dimension of x.
            +  What this does is flatten all dimensions of x except the last one, so if x was a 3D tensor 
                (e.g., with shape [batch_size, sequence_length, features]), it becomes a 
                2D tensor with shape [batch_size * sequence_length, features].

        y.contiguous().view(-1)
            + y.contiguous(): Similar to x.contiguous(), it ensures that the tensor y is stored contiguously in memory.
            + y.view(-1):
                + This flattens y into a 1D tensor (a single long list of values).
                + If y was originally a 2D tensor with shape [batch_size, sequence_length], 
                    it now becomes a 1D tensor with shape [batch_size * sequence_length].
    '''

    def __init__(self, generator: Generator, criterion: LabelSmoothing):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)

        # This comparison (via criterion) gives you the loss.
        raw_loss = self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )

        # The loss is then normalized by dividing it by norm. 
        #   Normalization helps in stabilizing the training process, especially when dealing with different batch sizes.
        #   It might be used to update the model during training.
        sloss = raw_loss / norm
 
        return raw_loss, sloss
