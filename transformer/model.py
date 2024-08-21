import torch.nn as nn

from transformer.mha import MultiHeadedAttention
from transformer.ffn import PositionWiseFeedForward
from transformer.penc import PositionalEncoding
from transformer.encdenc import EncoderDecoder
from transformer.encoder import Encoder, EncoderLayer
from transformer.decoder import Decoder, DecoderLayer
from transformer.embedding import Embeddings
from transformer.generator import Generator


import copy


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    '''
    What nn.Sequential Does? `nn.Sequential(Embeddings(d_model, src_vocab), c(position))`
        + Combines Modules: nn.Sequential allows you to group a sequence of layers or operations 
            that should be applied one after another. In your example, this means that the output 
            of the Embeddings layer will be directly passed as input to the c(position) layer.
        + Streamlines the Forward Pass: By using nn.Sequential, you don't need to manually define 
            the forward pass for each component. Instead, the forward pass through the sequence is 
            automatically handled, simplifying your code.
    '''
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
