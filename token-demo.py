import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Force Transformers to use PyTorch only

# Patch Transformers utility globally to avoid TensorFlow tensor checks.
from transformers.utils import generic
generic.is_tf_tensor = lambda x: False

import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GPT2Tokenizer

def extract_samples(output_filename="concatenated_samples.txt", num_samples=10):
    """
    Loads a small streaming subset of the FineWeb dataset ("sample-10BT"),
    extracts the first `num_samples` examples, concatenates their text fields,
    and writes the result to the specified output file.
    This function always downloads the dataset and writes fresh output.
    """
    print("Extracting samples from the dataset...")
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    concatenated_text = ""
    count = 0

    for sample in dataset:
        concatenated_text += sample["text"] + "\n\n"
        count += 1
        if count >= num_samples:
            break

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(concatenated_text)
    
    print(f"{num_samples} samples have been concatenated and saved to '{output_filename}'.")


def convert_text_to_utf8_byte_values(input_filename="concatenated_samples.txt", output_filename="utf8_bytes.txt"):
    """
    Reads text from the input file, encodes it to UTF-8 (yielding bytes),
    converts each byte to its integer value, and writes the numbers (space-separated)
    to the specified output file.
    """
    with open(input_filename, "r", encoding="utf-8") as f:
        text = f.read()
    
    byte_values = list(text.encode("utf-8"))
    number_str = " ".join(map(str, byte_values))
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(number_str)
    
    print(f"Text from '{input_filename}' has been converted to UTF-8 numbers and saved to '{output_filename}'.")


def convert_text_to_utf8_bit_strings(input_filename="concatenated_samples.txt", output_filename="utf8_bits.txt"):
    """
    Reads text from the input file, encodes it to UTF-8 (yielding bytes),
    converts each byte to its 8-bit binary representation,
    and writes the resulting bit strings (space-separated) to the specified output file.
    """
    with open(input_filename, "r", encoding="utf-8") as f:
        text = f.read()
    
    byte_values = text.encode("utf-8")
    bits = [format(b, '08b') for b in byte_values]
    bit_str = " ".join(bits)
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(bit_str)
    
    print(f"Text from '{input_filename}' has been converted to UTF-8 bits and saved to '{output_filename}'.")


def convert_text_to_tokens(input_filename="concatenated_samples.txt", output_filename="tokens.txt"):
    """
    Reads text from the input file, tokenizes it using GPT2's tokenizer,
    and writes the resulting token IDs (space-separated) to the specified output file.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    with open(input_filename, "r", encoding="utf-8") as f:
        text = f.read()
    
    token_ids = tokenizer.encode(text)
    token_str = " ".join(map(str, token_ids))
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(token_str)
    
    print(f"Text from '{input_filename}' has been tokenized using GPT2 and saved to '{output_filename}'.")


def convert_tokens_to_text(token_string):
    """
    Converts a space-separated string of GPT2 token IDs into a list of integers,
    decodes them back into text using GPT2's tokenizer,
    and prints the decoded text to the command line.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    try:
        token_ids = [int(x) for x in token_string.strip().split()]
    except ValueError:
        print("Error: Token string must contain space-separated integers.")
        sys.exit(1)
    
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    print("Decoded text:")
    print(decoded_text)


def predict_next_token(token_string, top_k=5):
    """
    Given a space-separated string of GPT2 token IDs,
    this function directly converts them into a tensor,
    uses GPT2 to predict the next token, and prints the top_k
    predictions along with their probability distribution.
    """
    try:
        token_ids = [int(x) for x in token_string.strip().split()]
    except ValueError:
        print("Error: token_string must be a space-separated list of integers.")
        sys.exit(1)
    
    # Directly create a tensor from the token IDs.
    input_ids = torch.tensor([token_ids])
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits  # shape: (batch_size, seq_length, vocab_size)
    next_token_logits = logits[0, -1, :]
    
    probabilities = torch.softmax(next_token_logits, dim=0)
    top_prob, top_indices = torch.topk(probabilities, top_k)
    top_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_indices]
    
    print("Top {} predictions for the next token:".format(top_k))
    for token, prob in zip(top_tokens, top_prob):
        print(f"Token: '{token}' - Probability: {prob.item():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python token-demo.py extract_samples")
        print("  python token-demo.py convert_text_to_utf8_byte_values")
        print("  python token-demo.py convert_text_to_utf8_bit_strings")
        print("  python token-demo.py convert_text_to_tokens")
        print('  python token-demo.py convert_tokens_to_text "91 860 287 11579 3962 5659 ..."')
        print('  python token-demo.py predict_next_token "91 860 287 11579 3962 5659 25 57049 28257" [top_k]')
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "extract_samples":
        extract_samples()
    elif command == "convert_text_to_utf8_byte_values":
        convert_text_to_utf8_byte_values()
    elif command == "convert_text_to_utf8_bit_strings":
        convert_text_to_utf8_bit_strings()
    elif command == "convert_text_to_tokens":
        convert_text_to_tokens()
    elif command == "convert_tokens_to_text":
        if len(sys.argv) < 3:
            print("Error: Please provide a string of token IDs as a parameter.")
            print('Example: python token-demo.py convert_tokens_to_text "91 860 287 11579 3962 5659"')
            sys.exit(1)
        token_string = " ".join(sys.argv[2:])
        convert_tokens_to_text(token_string)
    elif command == "predict_next_token":
        if len(sys.argv) < 3:
            print("Error: Please provide a string of token IDs as a parameter.")
            print('Example: python token-demo.py predict_next_token "91 860 287 11579 3962 5659 25 57049 28257"')
            sys.exit(1)
        token_string = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
        predict_next_token(token_string, top_k)
    else:
        print(f"Unknown command '{command}'.")
        print("Available commands: extract_samples, convert_text_to_utf8_byte_values, convert_text_to_utf8_bit_strings, convert_text_to_tokens, convert_tokens_to_text, predict_next_token")
        sys.exit(1)
