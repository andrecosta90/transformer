from transformer.generator import Generator
from transformer.smoothing import LabelSmoothing


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator: Generator, criterion: LabelSmoothing):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
