import torch, torch.nn as nn, torch.nn.functional as F

SEMANTIC_VOCAB = 16_384          # MoonCast
DAC_VOCAB      = 1_024 * 9 + 2   # 9 bands + <eos> + <pad>
EOS            = DAC_VOCAB - 2
PAD            = DAC_VOCAB - 1

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pos = torch.arange(max_len)[:, None]
        i   = torch.arange(0, d_model, 2)[None]
        div = torch.exp(i * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe  = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe[None])

    def forward(self, x):          # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]

class Semantic2DAC(nn.Module):
    def __init__(self, d_model=384, n_layers=6, n_heads=12, dropout=0.05):
        super().__init__()
        self.src_emb = nn.Embedding(SEMANTIC_VOCAB, d_model)
        self.tgt_emb = nn.Embedding(DAC_VOCAB,      d_model)
        self.pos     = PositionalEncoding(d_model)

        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
        )
        self.lm_head = nn.Linear(d_model, DAC_VOCAB, bias=False)

    def forward(self, src, tgt):
        src = self.pos(self.src_emb(src))
        tgt = self.pos(self.tgt_emb(tgt[:, :-1]))          # teacher forcing
        out = self.tf(src, tgt)
        return self.lm_head(out)                           # [B, T-1, V]

    @torch.no_grad()
    def generate(self, src, max_multiplier=1.7):
        """Greedy decoding; src shape (T,)"""
        self.eval()
        src = src.unsqueeze(0).to(next(self.parameters()).device)
        mem = self.tf.encoder(self.pos(self.src_emb(src)))

        tgt = torch.full((1, 1), EOS, dtype=torch.long, device=src.device)
        for _ in range(int(src.size(1)*max_multiplier)+32):
            out = self.tf.decoder(self.pos(self.tgt_emb(tgt)), mem)
            logits = self.lm_head(out)[:, -1]              # last step
            next_token = logits.argmax(-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == EOS:
                break
        seq = tgt[0, 1:-1] if tgt[0, -1] == EOS else tgt[0, 1:]
        
        # Pad sequence to nearest multiple of 9
        seq_len = len(seq)
        pad_len = (9 - (seq_len % 9)) % 9  # Calculate padding needed
        if pad_len > 0:
            seq = torch.cat([seq, torch.full((pad_len,), PAD, dtype=seq.dtype, device=seq.device)])
        
        return seq.view(-1, 9).cpu().numpy()               # (L, 9) 