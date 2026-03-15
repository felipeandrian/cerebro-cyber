# example_bpe_encdec.py
# --- FUNÇÕES AUXILIARES (As mesmas de antes) ---
def obter_pares(ids):
    pares = {}
    for par in zip(ids, ids[1:]):
        pares[par] = pares.get(par, 0) + 1
    return pares

def fundir(ids, par_alvo, novo_id):
    novos_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == par_alvo[0] and ids[i+1] == par_alvo[1]:
            novos_ids.append(novo_id)
            i += 2
        else:
            novos_ids.append(ids[i])
            i += 1
    return novos_ids

# --- 1. O TREINAMENTO (Salvando o conhecimento) ---
texto_treino = "def seguranca():\n    print('ciberseguranca e seguranca de rede')"
tokens_treino = list(texto_treino.encode("utf-8"))

# Criamos as "memórias" da nossa IA
merges = {} # Salva as regras: ex: (101, 102) -> 256
vocab = {i: bytes([i]) for i in range(256)} # Vocabulário base (0 a 255)

num_fusoes = 5
for i in range(num_fusoes):
    pares = obter_pares(tokens_treino)
    if not pares:
        break

    par_mais_comum = max(pares, key=pares.get)
    novo_id = 256 + i
    tokens_treino = fundir(tokens_treino, par_mais_comum, novo_id)

    # SALVANDO AS REGRAS PARA O ENCODE/DECODE!
    merges[par_mais_comum] = novo_id
    vocab[novo_id] = vocab[par_mais_comum[0]] + vocab[par_mais_comum[1]]

# --- 2. AS FUNÇÕES PRINCIPAIS ---

def decode(lista_ids):
    """ IA -> Humano (Pega os números e traduz para texto legível) """
    # Junta os bytes correspondentes a cada ID do nosso vocabulário
    bytes_juntos = b"".join([vocab[idx] for idx in lista_ids])
    # .decode("utf-8") transforma os bytes de máquina de volta em texto legível
    return bytes_juntos.decode("utf-8", errors="replace")

def encode(texto_usuario):
    """ Humano -> IA (Pega o texto, transforma em bytes e aplica a compressão) """
    ids = list(texto_usuario.encode("utf-8"))

    # Enquanto houver pelo menos 2 tokens para tentar fundir
    while len(ids) >= 2:
        pares = obter_pares(ids)

        # Procura no texto atual qual par tem a regra de fusão mais prioritária
        # (Aquela que foi aprendida primeiro no treinamento)
        par_encontrado = min(pares, key=lambda p: merges.get(p, float("inf")))

        # Se o par não está nas nossas regras salvas, não há mais o que comprimir
        if par_encontrado not in merges:
            break

        novo_id = merges[par_encontrado]
        ids = fundir(ids, par_encontrado, novo_id)

    return ids

# --- 3. TESTANDO NA PRÁTICA ---
print("--- TESTANDO ENCODE E DECODE ---")
texto_teste = "seguranca"

# Você digita o texto
ids_codificados = encode(texto_teste)
print(f"1. Você digitou: '{texto_teste}'")
print(f"2. Encode (O que a IA lê): {ids_codificados}")

# A IA devolve a resposta
texto_decodificado = decode(ids_codificados)
print(f"3. Decode (Traduzido de volta): '{texto_decodificado}'")

# Verificação de segurança
assert texto_teste == texto_decodificado, "Erro! O Decode não bateu com o original!"
print("\nSucesso absoluto! O Tokenizador está 100% funcional.")
