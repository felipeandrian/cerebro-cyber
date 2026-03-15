# tempo.py
import torch
import torch.nn as nn

# --- Nossas variáveis de ambiente ---
batch_size = 2    # Quantas frases lemos por vez
block_size = 4    # Tamanho da frase (janela de contexto)
vocab_size = 300  # Tamanho do dicionário BPE
n_embd = 32       # Dimensões de significado (Canais)

# --- 1. AS DUAS TABELAS DE MEMÓRIA ---
tabela_tokens = nn.Embedding(vocab_size, n_embd)
# A NOVIDADE: Uma tabela só para as posições (0, 1, 2 e 3)
tabela_posicoes = nn.Embedding(block_size, n_embd)

# Fingindo que este é o nosso lote (X) vindo do Dataloader
xb = torch.tensor([[10, 25, 88, 30],
                   [45, 99, 12, 55]])

# --- 2. EXTRAINDO O SIGNIFICADO E A ORDEM ---
# "Quem" são as palavras?
emb_tokens = tabela_tokens(xb) # Formato (B, T, C) -> (2, 4, 32)

# "Onde" elas estão?
# torch.arange(4) cria uma lista simples: [0, 1, 2, 3]
posicoes = torch.arange(block_size)
emb_posicoes = tabela_posicoes(posicoes) # Formato (T, C) -> (4, 32)

# --- 3. A FUSÃO MÁGICA ---
# Somamos o significado com a posição.
# A palavra "vírus" na posição 0 terá uma matemática diferente da palavra "vírus" na posição 3!
x_final = emb_tokens + emb_posicoes

print("--- DANDO NOÇÃO DE TEMPO PARA A IA ---")
print(f"Formato dos Tokens:   {emb_tokens.shape}")
print(f"Formato das Posições: {emb_posicoes.shape}")
print("-" * 40)
print(f"Formato Final (X):    {x_final.shape}")
print("\nO cérebro da nossa IA agora sabe O QUÊ e ONDE cada token está!")
