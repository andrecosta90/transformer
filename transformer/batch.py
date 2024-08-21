from transformer.mask import subsequent_mask


class Batch:
    """
    Object for holding a batch of data with mask during training.

    pad: The value used to represent padding tokens in the sequences. 
        The default value is 2, which might correspond to a special <blank> token in the vocabulary.

    self.src_mask: A mask that indicates where the padding tokens (pad) are in the source sequence. 
        The mask is expanded along the last dimension (using .unsqueeze(-2)) to fit 
        the expected shape for further processing.

    self.tgt: The target sequence excluding the last token (tgt[:, :-1]), used as input to the decoder.

    self.tgt_y: The target sequence excluding the first token (tgt[:, 1:]), used as the expected output 
        for training (i.e., what the model is trying to predict).

    self.tgt_mask: A mask for the target sequence that hides padding and future tokens, 
        created by calling self.make_std_mask.
        
    self.ntokens: The number of non-padding tokens in self.tgt_y, used for loss computation and normalization.
    """

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
