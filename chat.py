import torch
import tiktoken
from arquitetura import CerebroCyber, block_size

print("--- INICIANDO O SISTEMA CEREBROCYBER ---")
print("Carregando pesos neurais (2.8M tokens de sabedoria)...")

# 1. Preparação
tokenizador = tiktoken.get_encoding("cl100k_base")
modelo = CerebroCyber()
modelo.load_state_dict(torch.load('cerebro_cyber.pt'))
modelo.eval() # Modo de leitura travado (sem gastar RAM treinando)

# 2. Motor de Geração
def gerar(modelo, prompt, max_novos_tokens):
    idx = torch.tensor(tokenizador.encode(prompt)).unsqueeze(0)
    
    for _ in range(max_novos_tokens):
        # Corta para o tamanho da janela de contexto
        idx_cond = idx[:, -block_size:]
        
        with torch.no_grad(): # Economiza muita RAM na hora de falar
            logits, _ = modelo(idx_cond)
            
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        proximo_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, proximo_token), dim=1)
        
    return tokenizador.decode(idx[0].tolist())

print("\nSistema Online. Digite 'sair' para encerrar.")
print("-" * 50)

# 3. O Loop do Chat
while True:
    usuario = input("root@cerebrocyber:~# ")
    
    if usuario.lower() == 'sair':
        print("Desligando o sistema...")
        break
        
    if usuario.strip() == "":
        continue
        
    resposta = gerar(modelo, usuario, max_novos_tokens=60)
    print(f"\n[CerebroCyber]: {resposta}\n")
