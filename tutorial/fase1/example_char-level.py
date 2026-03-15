# example_char-level.py
# Charater-Level
# 1. Nosso "Dataset" inicial de exemplo
# É uma string simples misturando código e texto.
texto = "def analisar_malware():\n    print('Sistema seguro!')"

# 2. Criando o Vocabulário
# O comando set() pega todas as letras do texto, mas remove as repetidas.
# O comando sorted() organiza tudo em ordem alfabética para ficar organizado.
caracteres = sorted(list(set(texto)))
tamanho_vocabulario = len(caracteres)

print("--- PASSO 1: O VOCABULÁRIO ---")
print(f"Nossa IA só conhece estes {tamanho_vocabulario} caracteres:")
print("".join(caracteres))
print("-" * 30)

# 3. Criando os Tradutores (Os Dicionários)
# stoi: String TO Integer (Letra para Número)
# itos: Integer TO String (Número para Letra)
stoi = { char: i for i, char in enumerate(caracteres) }
itos = { i: char for i, char in enumerate(caracteres) }

# 4. A Função de Codificar (Humano -> IA)
def encode(texto_humano):
    # Passa por cada letra do texto e devolve o número correspondente
    return [stoi[letra] for letra in texto_humano]

# 5. A Função de Decodificar (IA -> Humano)
def decode(lista_numeros):
    # Passa por cada número da lista e devolve a letra correspondente
    return "".join([itos[numero] for numero in lista_numeros])

# --- TESTANDO A MÁGICA ---
print("\n--- PASSO 2: TESTANDO O TOKENIZADOR ---")

texto_teste = "def malware"

# Transformando em números
mensagem_codificada = encode(texto_teste)
print(f"1. Você digitou: '{texto_teste}'")
print(f"2. A IA 'enxerga' assim (Codificado): {mensagem_codificada}")

# Transformando de volta em texto
mensagem_decodificada = decode(mensagem_codificada)
print(f"3. A IA responde para você (Decodificado): '{mensagem_decodificada}'")
