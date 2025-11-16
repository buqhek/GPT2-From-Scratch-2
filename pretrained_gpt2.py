"""Test the GPT2 model with pretrained weight from HuggingFace Transformers."""
import torch
import torch.nn.functional as F
from gpt2 import GPT
import sys

def main():
    """If this code runs and produces good coherent text, then the model is working and we have nearly identically the same model as the HuggingFace Transformers GPT2 model."""
    # attempt to auto detect the device including cuda and mps and cpu
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
        # TODO: Implement training code/flag here

    if '--manual' in args:  # manual writing mode
        manual_mode = True
    exit(0)


    model = GPT.from_pretrained('gpt2')
    print("Model loaded successfully.")

    num_return_sequences = 1
    max_length = 30

    model.eval()
    model.to(device)

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