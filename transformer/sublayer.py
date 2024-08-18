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
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
