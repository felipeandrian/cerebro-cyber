import os
import tiktoken
import numpy as np
import fitz  # PyMuPDF

# Configurações
pasta_raiz = 'livros'
arquivo_saida = 'dataset_cyber.bin'
enc = tiktoken.get_encoding("cl100k_base")

print(f"--- ENGOLIDOR DE CONHECIMENTO (MODO RECURSIVO) ---")

todas_palavras_tokens = []
contador_arquivos = 0

# 1. os.walk percorre a pasta raiz, as subpastas e os arquivos
for raiz, pastas, arquivos in os.walk(pasta_raiz):
    for nome_arq in arquivos:
        if nome_arq.endswith(('.txt', '.pdf')):
            caminho_completo = os.path.join(raiz, nome_arq)
            texto = ""
            contador_arquivos += 1
            
            try:
                if nome_arq.endswith('.pdf'):
                    print(f"Extraindo PDF [{contador_arquivos}]: {nome_arq}...", end=" ")
                    doc = fitz.open(caminho_completo)
                    for pagina in doc:
                        texto += pagina.get_text()
                    doc.close()
                else:
                    print(f"Lendo TXT [{contador_arquivos}]: {nome_arq}...", end=" ")
                    with open(caminho_completo, 'r', encoding='utf-8', errors='ignore') as f:
                        texto = f.read()
                
                # Transforma em números e guarda
                tokens = enc.encode(texto)
                todas_palavras_tokens.extend(tokens)
                print(f"({len(tokens)} tokens)")
            
            except Exception as e:
                print(f"\n[ERRO] Falha ao ler {nome_arq}: {e}")

if contador_arquivos == 0:
    print("ERRO: Nenhum arquivo .txt ou .pdf encontrado!")
else:
    # 2. Salvando a massa bruta no HD
    print("-" * 30)
    print("Finalizando compressão...")
    dados_final = np.array(todas_palavras_tokens, dtype=np.int32)
    dados_final.tofile(arquivo_saida)

    print(f"SUCESSO! {contador_arquivos} arquivos processados em todas as pastas.")
    print(f"Total de tokens no dataset: {len(dados_final)}")
