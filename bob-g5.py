import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =========================
# CONFIG
# =========================
seq_len = 32      
emb_dim = 128     
heads = 4         
layers = 3        
ff_dim = 512      

epochs = 150      # More epochs for the bigger brain
batch_size = 32
lr = 3e-4
weight_decay = 0.01 

model_file = "bob_g5.pt" #   BRAND NEW FILE! MUST BE G5!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# LOAD DATA
# =========================
with open("training.txt", "r", encoding="utf-8") as f:
    lines = [line.strip().lower() for line in f if line.strip()]

words = []
for line in lines:
    words.extend(["<start>"] + line.split() + ["<end>"])

#   DATA MULTIPLIER! Tiny models need massive repetition to learn!
# We will repeat the dataset 5 times so Bob memorizes the facts.
words = words * 5

# =========================
# VOCAB
# =========================
vocab = ["<pad>", "<unk>"] + sorted(set(words))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

# =========================
# DATASET
# =========================
token_ids = [word_to_idx[w] for w in words]

X = []
Y = []
for i in range(0, len(token_ids) - seq_len - 1, seq_len):
    X.append(token_ids[i : i + seq_len])
    Y.append(token_ids[i + 1 : i + seq_len + 1])

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# =========================
# MODEL
# =========================
class BobGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Embedding(seq_len, emb_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_embed(x) + self.pos_embed(positions)
        
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        x = self.transformer(x, mask=mask)
        x = self.ln(x)
        return self.head(x)

# =========================
# CREATE MODEL
# =========================
model = BobGPT().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}") #     THIS MUST SAY 1,364,823 !    

# Do NOT load old models. We want a fresh start.
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    print("Loaded existing Bob G5.")
else:
    print("Building brand new Bob G5 brain!")

# =========================
# TRAIN
# =========================
loss_fn = nn.CrossEntropyLoss(ignore_index=word_to_idx["<pad>"])
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

warmup_steps = 10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_steps)

print("Training Bob...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    if epoch < warmup_steps:
        lr_scale = (epoch + 1) / warmup_steps
        for pg in optimizer.param_groups:
            pg['lr'] = lr * lr_scale
            
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
    if epoch >= warmup_steps:
        scheduler.step()

    avg_loss = total_loss / len(loader)
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch} | Loss {avg_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}")

torch.save(model.state_dict(), model_file)
print("Bob saved.")

# =========================
# GENERATION (CLEANED + GREEDY + PENALTY)
# =========================
def generate(seed_words, max_new_tokens=20, repetition_penalty=1.5):
    model.eval()
    tokens = [word_to_idx.get(w, word_to_idx["<unk>"]) for w in seed_words]
    
    for _ in range(max_new_tokens):
        x = tokens[-seq_len:]
        x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(x)[0, -1]
            
        #   REPETITION PENALTY
        if len(tokens) > 1:
            recent_tokens = set(tokens[-5:]) 
            for tok_id in recent_tokens:
                word = idx_to_word[tok_id]
                if word not in ["<start>", "<end>", "<stop>", "<pad>", "<user>", "<bot>"]:
                    if logits[tok_id] > 0:
                        logits[tok_id] /= repetition_penalty
                    elif logits[tok_id] < 0:
                        logits[tok_id] *= repetition_penalty

        # Greedy Decoding
        next_id = torch.argmax(logits, dim=-1).item()
        
        if idx_to_word[next_id] in ["<end>", "<stop>"]:
            break
            
        tokens.append(next_id)
        
    #   FIX THE <UNK> BUG: Filter out <start> AND <unk> from the final output!
    result = [idx_to_word[t] for t in tokens if idx_to_word[t] not in ["<start>", "<end>", "<stop>", "<unk>", "<user>", "<bot>"]]
    return " ".join(result)

# =========================
# GENERATION (PURE GREEDY + HARD LOOP BLOCK)
# =========================
def generate(seed_words, max_new_tokens=20):
    model.eval()
    tokens = [word_to_idx.get(w, word_to_idx["<unk>"]) for w in seed_words]
    
    for _ in range(max_new_tokens):
        x = tokens[-seq_len:]
        x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(x)[0, -1]
            
        #   HARD LOOP BLOCK: Prevent A A A A loops
        # If the last token we generated is a normal word, ban it completely
        # so he CANNOT repeat it. But don't mess with his other math.
        if len(tokens) > 1:
            last_word = idx_to_word[tokens[-1]]
            if last_word not in ["<start>", "<end>", "<stop>", "<pad>", "<user>", "<bot>"]:
                logits[tokens[-1]] = float('-inf') # Total ban on the immediate last word

        #   PURE GREEDY: Pick the absolute highest probability word. No rolling dice.
        next_id = torch.argmax(logits, dim=-1).item()
        
        # Stop if he predicts the end
        if idx_to_word[next_id] in ["<end>", "<stop>"]:
            break
            
        tokens.append(next_id)
        
    # Filter out special tags so the chat looks clean
    result = [idx_to_word[t] for t in tokens if idx_to_word[t] not in ["<start>", "<end>", "<stop>", "<unk>", "<user>", "<bot>"]]
    return " ".join(result)

# =========================
# INTERACTIVE MODE
# =========================
print("\nBob-G5 Ready")
print("Type 'quit' to exit\n")

while True:
    seed = input("You: ").lower().strip()
    if seed == "quit":
        break
        
    # Format input with <user>, <stop>, and <bot> tags
    words = seed.split()
    context = ["<start>", "<user>"] + words + ["<stop>", "<bot>"]
    
    response = generate(context)
    
    print("Bob:", response, "\n")