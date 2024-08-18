from transformer.norm import LayerNorm

import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.

    This code is different from the paper! In the paper is smth like "return self.norm(x + sublayer(x))".
        Why? See the reference bellow:
    [1] https://github.com/harvardnlp/annotated-transformer/issues/92
    [2] https://github.com/harvardnlp/annotated-transformer/issues/100

    A residual connection is a simple but powerful idea used in deep learning models, 
        particularly in neural networks like transformers and ResNets.
    Imagine you're trying to learn something complicated, like a new skill.
        Instead of starting from scratch every time you try, you might want to build on what you already know.
        A residual connection works in a similar way.
    In a neural network, data is passed through several layers, each trying to learn some aspect of the task.
        Sometimes, though, these layers might not learn as well as they should.
        A residual connection helps by adding the original input back into the output of a layer. 
        It's like saying, "If this layer doesn't learn much, at least I still have the original information."
    So, with a residual connection, the network learns something new (the layer's output), 
        but also keeps what it already knows (the original input). 
        This helps the network learn better and makes it easier to train very deep networks.

    In simple terms: it's like giving the neural network a shortcut to avoid getting lost in complicated learning steps, 
        ensuring it doesn't forget the basics.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
