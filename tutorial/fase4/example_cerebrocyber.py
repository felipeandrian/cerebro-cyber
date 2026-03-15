# example_cerebrocyber.py
# --- A SUPERCLASSE DO CEREBRO DA IA ---
class CerebroCyber(nn.Module):
    def __init__(self, vocab_size, n_embd, num_cabecas, num_blocos, block_size):
        super().__init__()

        # 1. As Tabelas de Memória (Significado e Ordem)
        self.tabela_tokens = nn.Embedding(vocab_size, n_embd)
        self.tabela_posicoes = nn.Embedding(block_size, n_embd)

        # 2. O Cérebro Profundo (A Torre de Blocos)
        self.blocos = nn.Sequential(*[BlocoTransformer(n_embd, num_cabecas) for _ in range(num_blocos)])

        # 3. O Acabamento Final
        self.norm_final = nn.LayerNorm(n_embd)
        self.boca_da_ia = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, alvos=None):
        B, T = idx.shape

        emb_tok = self.tabela_tokens(idx) # (B, T, C)
        emb_pos = self.tabela_posicoes(torch.arange(T, device=idx.device)) # (T, C)
        x = emb_tok + emb_pos

        x = self.blocos(x)
        x = self.norm_final(x)

        logits = self.boca_da_ia(x)

        if alvos is not None:
            B, T, C = logits.shape
            logits_esticados = logits.view(B * T, C)
            alvos_esticados = alvos.view(B * T)
            perda = F.cross_entropy(logits_esticados, alvos_esticados)
            return logits, perda
        else:
            return logits, None

# --- TESTANDO A SUPER MÁQUINA ---
block_size = 4

# CHAMAMOS O NOME NOVO NA HORA DE FABRICAR!
modelo = CerebroCyber(vocab_size=300, n_embd=32, num_cabecas=4, num_blocos=3, block_size=block_size)

idx_entrada = torch.tensor([[10, 25, 88, 30], [45, 99, 12, 55]]) # X (Lote 2, Tempo 4)
alvos_y = torch.randint(0, 300, (2, 4)) # Y

logits, perda = modelo(idx_entrada, alvos_y)

print("--- O MOTOR DO CEREBRO CYBER IA ESTÁ MONTADO E RODANDO ---")
print(f"Formato da Saída (Logits): {logits.shape}")
print(f"Erro Inicial (Loss):       {perda.item():.4f}")
print("A arquitetura da LM está 100% pronta para aprender!")
