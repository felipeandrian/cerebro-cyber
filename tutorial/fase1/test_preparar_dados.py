import tiktoken
import numpy as np
import os

print("--- O TRITURADOR DE LIVROS ---")

# Simulando o texto de um livro inteiro de cibersegurança
texto_livro = """
Capítulo 1: Introdução ao CerebroCyber.
O firewall é a primeira linha de defesa contra ataques de rede.
Hackers exploram vulnerabilidades de buffer overflow para ganhar acesso.
A criptografia protege os dados em trânsito e em repouso.
""" * 100 # Multiplicando por 100 para fingir que é um texto grande!

# 1. Ligando o nosso compressor BPE
tokenizador = tiktoken.get_encoding("cl100k_base")

# 2. Convertendo todo o livro em uma longa lista de números matemáticos
print("Triturando o texto em tokens...")
ids = tokenizador.encode(texto_livro)
print(f"Total de tokens gerados: {len(ids)}")

# 3. Empacotando em um formato hiper-compacto (inteiros de 32 bits)
# O Numpy converte a lista do Python em um bloco de bytes contínuo e super rápido
dados_compactados = np.array(ids, dtype=np.int32)

# 4. Salvando no HD (Esse é o arquivo que a IA vai ler depois!)
arquivo_saida = "dataset_cyber.bin"
dados_compactados.tofile(arquivo_saida)

print(f"Sucesso! Dados salvos no disco como '{arquivo_saida}'")
print(f"Tamanho do arquivo no HD: {os.path.getsize(arquivo_saida) / 1024:.2f} KB")
