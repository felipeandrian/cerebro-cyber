# head_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# (Lote=2, Tempo=4, Canais=32, Vocab=300)
B, T, C = 2, 4, 32
vocab_size = 300

# Fingindo que esta é a saída estabilizada do nosso Bloco Transformer
saida_do_bloco = torch.randn(B, T, C)

# O "Y" do Dataloader (O Gabarito com as respostas certas)
alvos_y = torch.randint(0, vocab_size, (B, T))

print("--- A BOCA DA IA E O PROFESSOR ---")

# 1. A CAMADA FINAL (Language Modeling Head)
# Converte os 32 pensamentos em 300 opções de palavras
camada_final = nn.Linear(C, vocab_size)

# 2. GERANDO AS PREVISÕES (Logits)
# Logits são as notas brutas que a IA dá para cada token do dicionário
logits = camada_final(saida_do_bloco)

# 3. CALCULANDO O ERRO (Cross-Entropy Loss)
# O PyTorch pede para "esticarmos" a matriz para calcular o erro da turma toda de uma vez
B, T, V = logits.shape
logits_esticados = logits.view(B * T, V)  # Junta as dimensões B e T
alvos_esticados = alvos_y.view(B * T)     # Estica o gabarito também

# A Função de Erro matemática (Onde a mágica do aprendizado acontece)
perda = F.cross_entropy(logits_esticados, alvos_esticados)

print(f"Formato da Previsão:   {logits.shape} -> (Lote, Tempo, Dicionário)")
print(f"A Nota de Erro (Loss): {perda.item():.4f}")
print("\nO objetivo de todo o treinamento é fazer esse número de Loss chegar o mais perto de zero possível!")
