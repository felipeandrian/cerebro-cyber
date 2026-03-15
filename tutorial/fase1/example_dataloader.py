# example_dataloader.py

import torch

# 1. Fingindo que este é o nosso livro inteiro já tokenizado (10 números)
dados = torch.tensor([10, 25, 88, 30, 45, 99, 12, 55, 77, 21])

# 2. As duas configurações mais importantes para a sua RAM
block_size = 4  # A Janela de Contexto (Quantos tokens a IA consegue "lembrar" por vez)
batch_size = 2  # O tamanho do Lote (Quantos recortes mandamos para a placa de vídeo juntos)

def pegar_lote():
    # torch.randint escolhe posições aleatórias no nosso texto para começar a ler
    # É como abrir o livro em uma página aleatória
    posicoes_iniciais = torch.randint(len(dados) - block_size, (batch_size,))

    # Recorta os pedaços X (o que a IA lê)
    x = torch.stack([dados[i : i + block_size] for i in posicoes_iniciais])

    # Recorta os pedaços Y (a resposta correta, que é 1 passo no futuro)
    y = torch.stack([dados[i + 1 : i + block_size + 1] for i in posicoes_iniciais])

    return x, y

# 3. Testando o nosso sistema digestivo
xb, yb = pegar_lote()

print("--- O QUE VAI PARA A MEMÓRIA DA IA ---")
print("Matriz X (Entradas):")
print(xb)
print("\nMatriz Y (Alvos a serem adivinhados):")
print(yb)
print("-" * 40)

# Mostrando exatamente como a IA vai treinar no primeiro recorte:
print("\nComo a IA enxerga o tempo:")
for tempo in range(block_size):
    contexto = xb[0, :tempo+1].tolist()
    alvo = yb[0, tempo].item()
    print(f"Quando a IA lê {contexto} ---> O alvo que ela deve prever é {alvo}")
