# despertar.py
import torch
import torch.nn as nn # A biblioteca de Redes Neurais (Neural Networks)

# 1. As variáveis que já tínhamos (Nossa RAM agradece)
batch_size = 2    # Lotes simultâneos
block_size = 4    # Janela de contexto
vocab_size = 300  # Tamanho do nosso dicionário BPE (quantos tokens existem)

# 2. A NOVA VARIÁVEL: Tamanho do Cérebro
# Quantas "dimensões de significado" cada palavra vai ter?
# O GPT-3 usa 12288. Como nosso PC é modesto, vamos usar 32!
n_embd = 32

# 3. Criando a Camada de Embedding
# É literalmente uma tabela de consulta (Look-up Table)
camada_embedding = nn.Embedding(vocab_size, n_embd)

# 4. Simulando a entrada do nosso Dataloader (X)
# Lote de 2 sequências, cada uma com 4 tokens
xb = torch.tensor([[10, 25, 88, 30],
                   [45, 99, 12, 55]])

# 5. A MÁGICA ACONTECE AQUI: Passando o texto pela camada
# Transformamos os rótulos "burros" em vetores "inteligentes"
x_inteligente = camada_embedding(xb)

print("--- O DESPERTAR DO CÉREBRO ---")
print(f"Formato da entrada (X): {xb.shape} -> (Lote, Tempo)")
print(f"Formato da saída:       {x_inteligente.shape} -> (Lote, Tempo, Canais de Significado)")
print("-" * 40)
print(f"O token '10' (primeira palavra) agora é este vetor matemático:\n{x_inteligente[0, 0, :5]}... (continua até ter 32 números)")
