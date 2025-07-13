import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformer import build_transformer
from data_and_preprocessing import src_batch, tgt_batch, label_batch, src_mask, tgt_mask
import data_and_preprocessing
src_vocab_size = 52
tgt_vocab_size = 49
model=build_transformer(src_vocab_size, tgt_vocab_size,10,10, d_model=512, N=6, h=8, dropout=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, pad_token_id):
    model.train()
    total_loss = 0
    total_tokens = 0

    for src_batch, tgt_batch, label_batch in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        label_batch = label_batch.to(device)

        # Create masks
        src_mask = (src_batch == pad_token_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, src_len]
        tgt_pad_mask = (tgt_batch == pad_token_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, tgt_len]
        tgt_len = tgt_batch.size(1)
        look_ahead_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask | look_ahead_mask  # [B, 1, tgt_len, tgt_len]

        # Forward pass
        output = model(src_batch, tgt_batch, src_mask, tgt_mask)  # [B, tgt_len, vocab_size]

        # Compute loss (flattened)
        output = output.view(-1, output.size(-1))  # [B*tgt_len, vocab_size]
        label_batch = label_batch.view(-1)         # [B*tgt_len]

        loss = loss_fn(output, label_batch)  # [B*tgt_len]

        # Mask out padding in loss
        non_pad_mask = (label_batch != pad_token_id)
        loss = (loss * non_pad_mask).sum()
        total_tokens += non_pad_mask.sum().item()
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_tokens

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(reduction='none',ignore_index=1)
dataset = TensorDataset(src_batch, tgt_batch, label_batch)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

def train(model, dataloader, optimizer, loss_fn, pad_token_id, device, epochs=10):
    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device, pad_token_id)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")


train(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    pad_token_id=1,
    device=device,
    epochs=50  
)

