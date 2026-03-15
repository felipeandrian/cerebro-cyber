# treinamentoAdamW.py
import torch.optim as optim

# 1. LIGANDO O OTIMIZADOR (O nosso Professor matemático)
# AdamW é o padrão ouro da indústria hoje em dia. lr = Learning Rate (Taxa de aprendizado)
otimizador = optim.AdamW(modelo.parameters(), lr=0.01)

# Vamos criar uma única "frase" fixa para testar se ele consegue decorar
# (Lote de 1 exemplo, Tempo de 4 tokens)
xb = torch.tensor([[10, 25, 88, 30]]) # A pergunta
yb = torch.tensor([[25, 88, 30, 45]]) # O gabarito

print("--- INICIANDO A ACADEMIA DO CEREBROCYBER ---")
passos_de_treino = 100

for passo in range(passos_de_treino):

    # 1. O aluno tenta adivinhar a resposta
    logits, perda = modelo(xb, yb)

    # 2. O professor zera as anotações da rodada anterior
    otimizador.zero_grad()

    # 3. O professor calcula as derivadas de onde a IA errou (Backpropagation)
    perda.backward()

    # 4. A IA ajusta seus neurônios para não cometer o mesmo erro
    otimizador.step()

    # 5. Mostramos o progresso a cada 20 passos
    if passo % 20 == 0:
        print(f"Passo {passo:3d} | Nota de Erro (Loss): {perda.item():.4f}")

# Imprimimos o resultado final
print(f"Passo 100 | Nota de Erro (Loss): {perda.item():.4f}")
print("\nSe o erro caiu drasticamente, o seu motor está aprendendo de verdade!")
