# boxLLM.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- 1. A CAIXA DE FERRAMENTAS ---

class Head(nn.Module):
    def __init__(self, n_embd, tamanho_cabeca, block_size): 
        super().__init__()
        self.key = nn.Linear(n_embd, tamanho_cabeca, bias=False)
        self.query = nn.Linear(n_embd, tamanho_cabeca, bias=False)
        self.value = nn.Linear(n_embd, tamanho_cabeca, bias=False)
        self.register_buffer('mascara', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        pesos = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        pesos = pesos.masked_fill(self.mascara[:T, :T] == 0, float('-inf'))
        pesos = F.softmax(pesos, dim=-1)
        v = self.value(x)
        return pesos @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_cabecas, tamanho_cabeca, block_size):
        super().__init__()
        self.cabecas = nn.ModuleList([Head(n_embd, tamanho_cabeca, block_size) for _ in range(num_cabecas)])
        self.misturador = nn.Linear(num_cabecas * tamanho_cabeca, n_embd)

    def forward(self, x):
        relatorios_juntos = torch.cat([cabeca(x) for cabeca in self.cabecas], dim=-1)
        return self.misturador(relatorios_juntos)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    def forward(self, x):
        return self.net(x)

class BlocoTransformer(nn.Module):
    def __init__(self, n_embd, num_cabecas, block_size):
        super().__init__()
        tamanho_cabeca = n_embd // num_cabecas
        self.atencao = MultiHeadAttention(n_embd, num_cabecas, tamanho_cabeca, block_size)
        self.ffwd = FeedForward(n_embd)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.atencao(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x

class CerebroCyber(nn.Module):
    def __init__(self, vocab_size, n_embd, num_cabecas, num_blocos, block_size):
        super().__init__()
        self.tabela_tokens = nn.Embedding(vocab_size, n_embd)
        self.tabela_posicoes = nn.Embedding(block_size, n_embd)
        self.blocos = nn.Sequential(*[BlocoTransformer(n_embd, num_cabecas, block_size) for _ in range(num_blocos)])
        self.norm_final = nn.LayerNorm(n_embd)
        self.boca_da_ia = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, alvos=None):
        B, T = idx.shape
        emb_tok = self.tabela_tokens(idx)
        emb_pos = self.tabela_posicoes(torch.arange(T, device=idx.device))
        x = emb_tok + emb_pos
        x = self.blocos(x)
        x = self.norm_final(x)
        logits = self.boca_da_ia(x)

        if alvos is not None:
            B, T, C = logits.shape
            logits_esticados = logits.view(B * T, C)
            alvos_esticados = alvos.view(B * T)
            perda = F.cross_entropy(logits_esticados, alvos_esticados)
            return logits, perda
        else:
            return logits, None

# --- 2. PREPARANDO OS DADOS E TREINANDO ---
texto_real = "o firewall bloqueou o ataque do hacker no servidor"
letras_unicas = sorted(list(set(texto_real)))
vocab_size_real = len(letras_unicas)

mapa_pra_numero = {letra: i for i, letra in enumerate(letras_unicas)}
encode = lambda string: [mapa_pra_numero[c] for c in string]

dados_reais = torch.tensor(encode(texto_real))
block_size = 8

x_real = dados_reais[0 : block_size].unsqueeze(0)
y_real = dados_reais[1 : block_size + 1].unsqueeze(0)

# Criando a IA blindada
modelo_real = CerebroCyber(vocab_size=vocab_size_real, n_embd=32, num_cabecas=4, num_blocos=3, block_size=block_size)
otimizador = optim.AdamW(modelo_real.parameters(), lr=0.01)

print("--- TREINANDO O CEREBROCYBER ---")
for passo in range(100):
    logits, perda = modelo_real(x_real, y_real)
    otimizador.zero_grad()
    perda.backward()
    otimizador.step()

    if passo % 20 == 0:
        print(f"Treino {passo:3d} | Erro (Loss): {perda.item():.4f}")

print(f"Treino 100 | Erro Final: {perda.item():.4f}")
