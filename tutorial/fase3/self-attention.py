# self-attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Nossas variáveis (Lote, Tempo, Canais)
B, T, C = 2, 4, 32

# Fingindo que este é o nosso X final (Tokens + Posições) do passo anterior
x = torch.randn(B, T, C)

print("--- CONSTRUINDO O SELF-ATTENTION ---")

# 1. Criando os "Roteadores" Lineares (Eles vão gerar o Q, K e V para cada palavra)
head_size = 16 # Tamanho do nosso vetor de atenção
key_layer = nn.Linear(C, head_size, bias=False)
query_layer = nn.Linear(C, head_size, bias=False)
value_layer = nn.Linear(C, head_size, bias=False)

# 2. Cada palavra gera sua Pergunta (Q) e sua Identidade (K)
k = key_layer(x)   # Formato: (B, T, 16)
q = query_layer(x) # Formato: (B, T, 16)

# 3. Calculando o "Match" (Multiplicação de Matrizes)
# Multiplicamos as Queries pelas Keys (transpostas) e dividimos pela raiz quadrada para estabilizar
pesos = q @ k.transpose(-2, -1) * (head_size ** -0.5)

# 4. A MÁSCARA DO FUTURO (A Regra de Ouro)
# Criamos um triângulo de zeros e uns. Onde for 0, nós substituímos por -Infinito.
mascara_triangular = torch.tril(torch.ones(T, T))
pesos = pesos.masked_fill(mascara_triangular == 0, float('-inf'))

# Aplicamos a função Softmax (O -infinito vira 0%, e o resto vira uma porcentagem de atenção)
pesos = F.softmax(pesos, dim=-1)

# 5. A Troca de Informação (Values)
v = value_layer(x)
saida_atencao = pesos @ v # As palavras finalmente conversam!

print("Matriz de Atenção (Quem presta atenção em quem?):")
print(pesos[0]) # Mostrando as porcentagens do primeiro lote
print("-" * 40)
print(f"Formato final após a comunicação: {saida_atencao.shape}")
