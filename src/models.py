"""Many-to-one recurrent encoders over observed words only."""
import torch
from torch import nn


class NextWordModel(nn.Module):
    def __init__(self, kind, vocab_size, embedding_dim=64, representation_dim=96, dropout=0.15):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        hidden = representation_dim // 2 if kind == "birnn" else representation_dim
        if kind == "birnn" and representation_dim % 2:
            raise ValueError("Bidirectional representation_dim must be even")
        cls = {"rnn": nn.RNN, "birnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        kwargs = {"nonlinearity": "tanh"} if kind in ["rnn", "birnn"] else {}
        self.recurrent = cls(embedding_dim, hidden, batch_first=True,
                             bidirectional=(kind == "birnn"), **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(representation_dim, vocab_size)

    def forward(self, observed_words):
        # No target is accepted. Both directions end within the observed window.
        _, state = self.recurrent(self.embedding(observed_words))
        h = state[0] if self.kind == "lstm" else state
        representation = torch.cat([h[-2], h[-1]], dim=1) if self.kind == "birnn" else h[-1]
        return self.output(self.dropout(representation))


def from_config(kind, vocab_size, config):
    return NextWordModel(kind, vocab_size, config["embedding_dim"],
                         config["representation_dim"], config["dropout"])
