from bpe import VOCAB_SIZE
from torch import nn

from .attention import AttnBlock
from .utils import LayerNorm, Lin, D_MODEL, p, NUM_LAYERS


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = p(VOCAB_SIZE, D_MODEL, 2e-2)
        self.attn_blocks = nn.ModuleList(AttnBlock() for _ in range(NUM_LAYERS))
        self.proj = Lin(D_MODEL, VOCAB_SIZE)
        self.final_ln = LayerNorm()

    def forward(self, input):
        input = self.emb[input]
        res = input
        for l in self.attn_blocks:
            res = l(res)
        return self.proj(self.final_ln(res))


BATCH_SIZE = 2**10
loss = nn.CrossEntropyLoss()
