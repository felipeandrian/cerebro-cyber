# example_bpe.py
# BPE Byte Pair Encoding
# 1. O nosso texto de treinamento (focado no nosso nicho)
texto = "def seguranca():\n    print('ciberseguranca e seguranca de rede')"

# Transformamos o texto em uma lista de números de 0 a 255 (Bytes do UTF-8)
tokens = list(texto.encode("utf-8"))
print(f"Texto original em bytes (Tamanho: {len(tokens)} tokens):")
print(tokens)
print("-" * 50)

# 2. Função para contar os pares de tokens que aparecem juntos
def contar_pares(lista_tokens):
    contagem = {}
    # zip junta o token atual com o próximo (ex: token[0] e token[1])
    for par in zip(lista_tokens, lista_tokens[1:]):
        contagem[par] = contagem.get(par, 0) + 1
    return contagem

# 3. Função para fundir (merge) o par mais comum em um novo token
def fundir_tokens(lista_tokens, par_alvo, novo_id_token):
    novos_tokens = []
    i = 0
    while i < len(lista_tokens):
        # Se acharmos o par, trocamos pelo novo ID e pulamos 2 casas
        if i < len(lista_tokens) - 1 and lista_tokens[i] == par_alvo[0] and lista_tokens[i+1] == par_alvo[1]:
            novos_tokens.append(novo_id_token)
            i += 2
        else:
            # Se não, apenas copiamos o token atual e andamos 1 casa
            novos_tokens.append(lista_tokens[i])
            i += 1
    return novos_tokens

# 4. O TREINAMENTO DO BPE (Vamos fazer 5 fusões/merges)
num_fusoes = 5
novo_id = 256 # Os bytes vão até 255, então nossos novos tokens começam no 256

for i in range(num_fusoes):
    pares = contar_pares(tokens)
    if not pares:
        break # Se não tem mais pares, para

    # Pega o par que apareceu mais vezes
    par_mais_comum = max(pares, key=pares.get)

    # Substitui o par pelo novo ID
    tokens = fundir_tokens(tokens, par_mais_comum, novo_id)

    print(f"Fusão {i+1}: Juntamos o par {par_mais_comum} -> Virou o Token {novo_id}")
    novo_id += 1

print("-" * 50)
print(f"Texto comprimido (Novo tamanho: {len(tokens)} tokens):")
print(tokens)
