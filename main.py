# Train the simple copy task.

import torch
# cpu : 1m32s
# gpu : 0m38s battery saver
# gpu : 0m11s performance
torch.set_default_device("cuda")
print(torch.get_default_device()) 

from transformer.dummy import DummyOptimizer, DummyScheduler
from transformer.loss import SimpleLossCompute
from transformer.model import make_model
from transformer.optim import rate
from transformer.smoothing import LabelSmoothing
from transformer.utils import run_epoch, data_gen, greedy_decode


from torch.optim.lr_scheduler import LambdaLR



def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step,
            model_size=model.src_embed[0].d_model,
            factor=1.0,
            warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(10):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).cuda()
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len).cuda()
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


example_simple_model()
