# feed-foward.py

import torch
import torch.nn as nn

# Nossas variáveis de ambiente (Lote, Tempo, Canais)
B, T, C = 2, 4, 32

# Fingindo que esta é a saída da nossa "Reunião de Equipe" (O Multi-Head Attention)
saida_da_reuniao = torch.randn(B, T, C)

print("--- CONSTRUINDO O FEED-FORWARD ---")

# A MESA DE TRABALHO INDIVIDUAL
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # nn.Sequential roda os comandos em ordem, um após o outro
        self.net = nn.Sequential(
            # Passo 1: Expande a capacidade de pensamento em 4x (32 -> 128)
            nn.Linear(n_embd, 4 * n_embd),

            # Passo 2: A Função de Ativação (O "Neurônio" disparando)
            # O ReLU transforma qualquer número negativo em 0. É isso que
            # permite à IA aprender padrões complexos e não apenas linhas retas.
            nn.ReLU(),

            # Passo 3: Comprime de volta para o tamanho original (128 -> 32)
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

# Criando a rede e passando o nosso texto por ela
ffwd = FeedForward(n_embd=C)
reflexao_final = ffwd(saida_da_reuniao)

print(f"1. A palavra chegou da reunião com formato: {saida_da_reuniao.shape}")
print(f"2. A palavra refletiu e saiu com formato:   {reflexao_final.shape}")
print("\nPerfeito! O raciocínio individual foi concluído sem quebrar a matemática.")
