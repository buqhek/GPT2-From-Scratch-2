"""Test the GPT2 model with pretrained weight from HuggingFace Transformers."""
import torch
import torch.nn.functional as F
from gpt2 import GPT
from dataloader import DataLoaderLite
import sys

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Command Line Arguments
    args = sys.argv[1:]
    if '--train' in args:  # Train mode
        idx = args.index('--train', 0, len(args))

        if idx + 1 >= len(args):  # Error Handling
            print('No training file provided, exiting.')
            exit(0)

        print(f'Training file provided: {args[idx + 1]}')
        exit(1)
        train_file = args[idx + 1]
        with open(train_file, 'r') as train:
            train_file = train.read()
        training_mode = True
    elif '--manual' in args:  # manual writing mode
        manual_mode = True

    model = GPT.from_pretrained('gpt2')
    model.to(device)
    num_return_sequences = 1
    max_length = 30

    # Model Training Phase
    if training_mode:
        B, T = 8, 512
        train_loader = DataLoaderLite(B, T)
        epochs = 1

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for i in range(epochs):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {i+1}, Loss: {loss.item()}")

    model.eval()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    if manual_mode:
        while True:
            tokens = enc.encode(input("\nPlease enter a prompt for the GPT2 model:\n\n"))
            # tokens = enc.encode("Hello, I'm a large language model.")
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
            tokens = tokens.repeat(num_return_sequences, 1)  # Repeat for num_return_sequences
            x = tokens.to(device)  # Move to the same device as the model

            while x.size(1) < max_length:
                # forward the model
                with torch.no_grad():
                    logits = model(x)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
                    ix = torch.multinomial(topk_probs, num_samples=1)
                    xcol = torch.gather(topk_indices, dim=-1, index=ix)
                    x = torch.cat((x, xcol), dim=1)

            # Decode the generated tokens
            for i in range(num_return_sequences):
                tokens = x[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(">", decoded)
    else:
        tokens = enc.encode("Hello, I'm a large language model.")
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        tokens = tokens.repeat(num_return_sequences, 1)  # Repeat for num_return_sequences
        x = tokens.to(device)  # Move to the same device as the model

        while x.size(1) < max_length:
            # forward the model
            with torch.no_grad():
                logits = model(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1)
                xcol = torch.gather(topk_indices, dim=-1, index=ix)
                x = torch.cat((x, xcol), dim=1)

        # Decode the generated tokens
        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(">", decoded)

if __name__ == "__main__":
    main()
    print("Nothing crashed, everything is fine.")