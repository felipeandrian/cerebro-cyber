# treinar.py
import torch
import torch.optim as optim
import numpy as np
# Importamos a classe mestre e as configurações do motor
from arquitetura import CerebroCyber, n_embd, block_size, vocab_size

# --- CONFIGURAÇÕES DE TREINO ---
batch_size = 8
max_iters = 5000 # Agora podemos treinar por mais tempo!
learning_rate = 3e-3

# --- 1. CONEXÃO COM O DISCO (O Canudo) ---
dados_no_disco = np.memmap('dataset_cyber.bin', dtype=np.int32, mode='r')

def pegar_lote():
    limite = len(dados_no_disco) - block_size - 1
    ix = torch.randint(0, limite, (batch_size,))
    x = torch.stack([torch.from_numpy((dados_no_disco[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((dados_no_disco[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y

# --- 2. INICIALIZAÇÃO ---
modelo = CerebroCyber()
# Se você já tiver um treino anterior e quiser continuar de onde parou:
# modelo.load_state_dict(torch.load('cerebro_cyber.pt'))

otimizador = optim.AdamW(modelo.parameters(), lr=learning_rate)

# --- 3. LOOP DE TREINAMENTO ---
print("Treinando o CerebroCyber...")
for iter in range(max_iters):
    xb, yb = pegar_lote()
    logits, loss = modelo(xb, yb)

    otimizador.zero_grad(set_to_none=True)
    loss.backward()
    otimizador.step()

    if iter % 100 == 0:
        print(f"Passo {iter} | Erro: {loss.item():.4f}")

# --- 4. SALVAMENTO FINAL ---
torch.save(modelo.state_dict(), 'cerebro_cyber.pt')
print("Cérebro atualizado e salvo!")
