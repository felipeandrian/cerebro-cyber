#  CerebroCyber: Construindo um LLM do Zero

Este repositório documenta a minha jornada educacional de engenharia reversa e construção de um **Modelo de Linguagem (Language Model)** baseado na arquitetura Transformer (Decoder-Only), a mesma tecnologia fundamental por trás do GPT, Llama e Claude.

O objetivo deste projeto não foi usar APIs prontas ou caixas pretas, mas sim entender a matemática, a engenharia de dados e a arquitetura de redes neurais profundas escrevendo cada linha de código em PyTorch.

##  Arquitetura do Motor

O CerebroCyber foi construído com as seguintes especificações e componentes:
* **Tokenização:** BPE (Byte-Pair Encoding) utilizando o padrão `cl100k_base` (mesmo vocabulário do GPT-4).
* **Embeddings:** Projeção dimensional de tokens e codificação posicional (Positional Encoding).
* **Masked Multi-Head Attention:** O núcleo de raciocínio, utilizando matrizes triangulares inferiores para garantir a previsão causal autoregressiva (a IA não pode "ver o futuro").
* **Feed-Forward Networks:** Expansão não-linear com ativação ReLU.
* **Estabilização:** Implementação da arquitetura Pre-Norm com `LayerNorm` e Conexões Residuais (estilo ResNet) para evitar o desaparecimento do gradiente.

##  O Que Eu Aprendi Construindo Isso

Construir essa IA do zero me permitiu dominar conceitos complexos do Machine Learning:
1. **Dataloaders de Nível Industrial:** Como lidar com terabytes de texto usando `numpy.memmap` sem estourar a memória RAM.
2. **Backpropagation e Otimização:** O uso do otimizador `AdamW`, taxas de aprendizado e o cálculo da Função de Perda de Entropia Cruzada (*Cross-Entropy Loss*).
3. **Mecânica da Geração de Texto:** A implementação do loop autoregressivo e o controle de "criatividade" usando distribuições de probabilidade (`torch.multinomial`).
4. **Gerenciamento de VRAM:** Otimizações como `torch.no_grad()` para inferência e `.to('cuda')` para aceleração em hardware.

##  Como Usar e Testar

### 1. Instalação
Clone o repositório e instale as dependências:
```bash
git clone [https://github.com/felipeandrian/cerebro-cyber.git](https://github.com/felipeandrian/cerebro-cyber.git)
cd cerebro-cyber
pip install -r requirements.txt

```

### 2. Refinando os Dados

Coloque seus arquivos `.txt` ou `.pdf` na pasta `livros/` e rode o extrator para gerar o combustível binário do modelo:

```bash
python preparar_dados.py

```

### 3. Treinando o Modelo

Inicie o loop de treinamento no dinamômetro. (Recomendado o uso de GPU via CUDA):

```bash
python treino.py

```

### 4. Conversando com a IA

Após o treinamento gerar o arquivo `.pt`, inicie o motor de inferência:

```bash
python chat.py

```

> *"O que eu não posso criar, eu não entendo."* — Richard Feynman

