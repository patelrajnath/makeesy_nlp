import spacy
import torch
from torch import nn
from torchtext.legacy import data
from torchtext.legacy.data import Field
from torchtext.legacy.datasets import Multi30k

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

MIN_FREQ = 2
SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
TRG.build_vocab(train_data.trg, min_freq=MIN_FREQ)

train_iter = data.Iterator(train_data, batch_size=1,
                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                           repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                           train=True)

input_dim = len(SRC.vocab.stoi)
output_dim = len(TRG.vocab.stoi)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers, dropout=0.01):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embeddings = nn.Embedding(self.input_dim, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.num_layers)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        tensor = self.embeddings(x)
        tensor = self.dropout_layer(tensor)

        outputs, (hidden, cell) = self.rnn(tensor)

        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers, dropouts=0.01):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dm = hid_dim
        self.num_layers = num_layers
        self.dropout = dropouts

        self.embeddings = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers)
        self.translate = nn.Linear(hid_dim, output_dim)

    def forward(self, target, hidden, cell):
        traget = target.unsqeez(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg=None):
        hidden, cell = self.encoder(src)
        bs, slen = src.size()

        print(hidden.size())


encoder = Encoder(input_dim, 16, 16, 1)
decoder = Decoder(output_dim, 16, 16, 1)
seq2seq = Seq2Seq(encoder, decoder)

for batch in train_iter:
    seq2seq(batch.src, batch.trg)
    exit()
