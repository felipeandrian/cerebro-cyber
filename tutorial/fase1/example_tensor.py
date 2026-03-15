# example_tensor.py

import torch

# Lembra da nossa frase codificada pelo BPE?
# Vamos fingir que o nosso Tokenizador cuspiu estes IDs:
ids_do_texto = [256, 14, 89, 257, 10]

# Transformando uma lista "burra" do Python em um Cérebro (Tensor)
dados = torch.tensor(ids_do_texto)

print("--- O PRIMEIRO PASSO DA IA ---")
print(f"Lista normal do Python: {ids_do_texto}")
print(f"Tensor do PyTorch:      {dados}")
print("-" * 30)
print(f"Formato (Shape): {dados.shape}")
print(f"Tipo de dado:    {dados.dtype}")
