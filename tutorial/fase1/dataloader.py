import torch
import numpy as np

print("--- CONSTRUINDO O CANUDO DE DADOS (MEMMAP) ---")

# 1. Conectando no arquivo no HD sem carregar ele na RAM!
# modo 'r' significa leitura (read).
arquivo_binario = 'dataset_cyber.bin'
dados_no_disco = np.memmap(arquivo_binario, dtype=np.int32, mode='r')

print(f"Arquivo conectado! Total de tokens disponíveis no HD: {len(dados_no_disco)}")

# 2. O nosso novo Dataloader super eficiente
def pegar_lote_do_disco(tamanho_lote, tamanho_bloco):
    # Sorteia posições aleatórias no livro (garantindo que não passe do final)
    limite = len(dados_no_disco) - tamanho_bloco - 1
    posicoes = torch.randint(0, limite, (tamanho_lote,))
    
    # Suga apenas os recortes exatos do HD para a RAM e converte para PyTorch
    # O astype(np.int64) garante que o PyTorch aceite os números do BPE
    x = torch.stack([torch.from_numpy((dados_no_disco[i : i + tamanho_bloco]).astype(np.int64)) for i in posicoes])
    y = torch.stack([torch.from_numpy((dados_no_disco[i + 1 : i + 1 + tamanho_bloco]).astype(np.int64)) for i in posicoes])
    
    return x, y

# 3. Testando o Motor de Busca
LOTE = 2   # Quantas frases ele lê ao mesmo tempo
TEMPO = 8  # Tamanho da janela de contexto (block_size)

xb, yb = pegar_lote_do_disco(LOTE, TEMPO)

print("\n--- TESTE DE SUCÇÃO ---")
print(f"Recorte X (O que a IA lê): Formato {xb.shape}")
print(f"Recorte Y (O gabarito):    Formato {yb.shape}")
print("A memória RAM está salva! Os dados vieram direto do disco.")
