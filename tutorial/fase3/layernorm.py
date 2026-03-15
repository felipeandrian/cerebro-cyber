# LayerNorm.py
import torch
import torch.nn as nn

# (Apenas relembrando nossas variáveis: Lote=2, Tempo=4, Canais=32)
B, T, C = 2, 4, 32

print("--- MONTANDO O BLOCO TRANSFORMER ---")

# Vamos empacotar tudo que fizemos até agora nesta super-classe
class BlocoTransformer(nn.Module):
    def __init__(self, n_embd, num_cabecas):
        super().__init__()
        # Dividimos os canais entre as cabeças (ex: 32 canais / 4 cabeças = 8)
        tamanho_cabeca = n_embd // num_cabecas

        # 1. A Reunião de Equipe (Que fizemos antes)
        self.atencao = MultiHeadAttention(num_cabecas, tamanho_cabeca)

        # 2. A Reflexão Individual (Que fizemos antes)
        self.ffwd = FeedForward(n_embd)

        # 3. OS FREIOS (Layer Normalization)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # A MÁGICA DO "ADD & NORM" (Pre-Norm architecture)

        # Passo 1: Normaliza -> Aplica Atenção -> Soma com o original (Bypass)
        x = x + self.atencao(self.norm1(x))

        # Passo 2: Normaliza -> Aplica Reflexão -> Soma com o original (Bypass)
        x = x + self.ffwd(self.norm2(x))

        return x

# --- TESTANDO O MOTOR COMPLETO ---
# Simulando nosso texto entrando
entrada_x = torch.randn(B, T, C)

# Criando 1 Bloco de Inteligência (com 4 cabeças)
bloco = BlocoTransformer(n_embd=C, num_cabecas=4)

# Passando o texto pelo bloco
saida_estabilizada = bloco(entrada_x)

print(f"Entrada (X) do bloco: {entrada_x.shape}")
print(f"Saída estabilizada:   {saida_estabilizada.shape}")
print("\nO coração da IA bateu! A matemática fluiu pela atenção, pela reflexão e saiu estabilizada.")
