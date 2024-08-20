import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    '''
    Positional encoding is a way to give a sense of "order" or "position" 
    to sequences of data, like words in a sentence, when using models that 
    don't naturally understand order, such as Transformers.

    Imagine you have a list of words, but they're just floating in space 
    without any order. If you want to know the first word, second word, etc.,
    you need a way to "tag" them with their positions.

    Positional encoding does this by adding a special set of numbers 
    to each word's data. These numbers aren't random; they're designed 
    in a specific way so that the model can figure out where each word 
    is in the sentence.

    Think of it like giving each word a unique "address" in the sentence, 
    so the model knows where it belongs, even if it doesn't process the 
    sentence in the usual left-to-right order. This helps the model 
    understand things like "the dog chased the cat" is different from 
    "the cat chased the dog" because the order of words matters.

    In summary, positional encoding is like adding GPS coordinates 
    to each word in a sentence so that the model knows where each word 
    is supposed to be.
    '''

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
