# multi-head-attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Nossas variáveis de ambiente
B, T, C = 2, 4, 32  # Batch(2), Tempo(4), Canais/Embedding(32)
x = torch.randn(B, T, C) # Nosso texto simulado

print("--- CONSTRUINDO O MULTI-HEAD ATTENTION ---")

# 1. A FÁBRICA DE CABEÇAS (A receita do que fizemos antes)
class Head(nn.Module):
    def __init__(self, tamanho_cabeca):
        super().__init__()
        self.key = nn.Linear(C, tamanho_cabeca, bias=False)
        self.query = nn.Linear(C, tamanho_cabeca, bias=False)
        self.value = nn.Linear(C, tamanho_cabeca, bias=False)
        # register_buffer salva a máscara triangular na memória da IA
        self.register_buffer('mascara', torch.tril(torch.ones(T, T)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Calcula a atenção
        pesos = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        pesos = pesos.masked_fill(self.mascara[:T, :T] == 0, float('-inf'))
        pesos = F.softmax(pesos, dim=-1)
        # Aplica os valores
        v = self.value(x)
        return pesos @ v

# 2. O GERENTE DA EQUIPE (O Multi-Head)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_cabecas, tamanho_cabeca):
        super().__init__()
        # Cria uma lista com as nossas 4 cabeças independentes
        self.cabecas = nn.ModuleList([Head(tamanho_cabeca) for _ in range(num_cabecas)])
        # Uma camada linear no final para misturar o relatório de todo mundo
        self.misturador = nn.Linear(num_cabecas * tamanho_cabeca, C)

    def forward(self, x):
        # Roda todas as cabeças ao mesmo tempo e "cola" (concatena) os resultados lado a lado
        relatorios_juntos = torch.cat([cabeca(x) for cabeca in self.cabecas], dim=-1)
        # Mistura tudo para a IA processar
        saida_final = self.misturador(relatorios_juntos)
        return saida_final

# 3. TESTANDO A EQUIPE
# Vamos criar 4 cabeças, cada uma processando 8 canais (4 * 8 = 32 canais no total)
mha = MultiHeadAttention(num_cabecas=4, tamanho_cabeca=8)

saida = mha(x)

print(f"Formato da Entrada (X): {x.shape}")
print(f"Formato da Saída:       {saida.shape}")
print("\nSucesso! As 4 cabeças leram o texto, conversaram e juntaram as conclusões no mesmo formato original (32 canais)!")
