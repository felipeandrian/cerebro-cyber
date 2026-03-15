import tiktoken

print("--- INSTALANDO O CHIP BPE OFICIAL ---")

# Carregando o dicionário hiper-otimizado (O mesmo do GPT-4)
tokenizador = tiktoken.get_encoding("cl100k_base")

# Vamos testar com uma frase complexa do nosso mundo de cibersegurança
texto_livro = "O CerebroCyber detectou uma vulnerabilidade de buffer overflow no firewall."

# Encode: Transformando o texto na linguagem matemática universal da IA
ids_bpe = tokenizador.encode(texto_livro)

print(f"Texto original: '{texto_livro}'")
print("-" * 40)
print(f"IDs gerados pelo BPE: {ids_bpe}")
print(f"Quantidade de tokens: {len(ids_bpe)}")

# Decode: Provando que a IA consegue destraduzir perfeitamente
texto_recuperado = tokenizador.decode(ids_bpe)
print("-" * 40)
print(f"Texto destraduzido: '{texto_recuperado}'")
