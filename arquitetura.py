import torch
import torch.nn as nn
import torch.nn.functional as F

# Constantes que o modelo usa
n_embd = 64
num_cabecas = 4
num_blocos = 6
block_size = 128
vocab_size = 100277 

class Head(nn.Module):
    def __init__(self, tamanho_cabeca):
        super().__init__()
        self.key = nn.Linear(n_embd, tamanho_cabeca, bias=False)
        self.query = nn.Linear(n_embd, tamanho_cabeca, bias=False)
        self.value = nn.Linear(n_embd, tamanho_cabeca, bias=False)
        self.register_buffer('mascara', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        pesos = q @ k.transpose(-2, -1) * (C ** -0.5)
        pesos = pesos.masked_fill(self.mascara[:T, :T] == 0, float('-inf'))
        return F.softmax(pesos, dim=-1) @ self.value(x)
    pass

class MultiHeadAttention(nn.Module):
    def __init__(self, num_cabecas, tamanho_cabeca):
        super().__init__()
        self.cabecas = nn.ModuleList([Head(tamanho_cabeca) for _ in range(num_cabecas)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.cabecas], dim=-1))
    pass

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x): return self.net(x)
    pass

class BlocoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(num_cabecas, n_embd // num_cabecas)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    pass

class CerebroCyber(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, n_embd)
        self.pos_emb_table = nn.Embedding(block_size, n_embd)
        self.blocos = nn.Sequential(*[BlocoTransformer() for _ in range(num_blocos)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None): # <-- O 'targets' precisa estar aqui!
        B, T = idx.shape
        
        # O pensamento da IA
        tok_emb = self.token_emb_table(idx)
        pos_emb = self.pos_emb_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocos(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Se não passarmos o gabarito (targets), ela só devolve os palpites
        if targets is None:
            return logits, None
        
        # Se passarmos o gabarito, ela calcula o erro (Loss) para o treino
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    pass
	
