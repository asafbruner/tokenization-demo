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


def predict_next_token(input_str, top_k=5, num_predict=1):
    """
    If num_predict == 1:
      Computes and prints the top_k predictions for the next token along with their probabilities.
    If num_predict > 1:
      Iteratively predicts one token at a time (using greedy decoding) and appends it to the current sequence.
      After each iteration, the function prints the updated sequence as token IDs and as decoded text.
      
    The input can be provided as a space-separated list of token numbers (e.g. "91 7680 278")
    or as plain text (e.g. "Hello, how are you?").
    """
    # Load tokenizer and model (using GPT-2 as an example)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    
    # Ensure pad_token is set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    # Try to interpret the input as token numbers.
    tokens_possible = input_str.strip().split()
    try:
        token_ids = [int(x) for x in tokens_possible]
        input_type = "token_ids"
    except ValueError:
        # If conversion fails, assume it's a plain text string.
        token_ids = tokenizer.encode(input_str)
        input_type = "text"
    
    # Convert token_ids list into a tensor (batch size 1)
    current_ids = torch.tensor([token_ids])
    
    if num_predict == 1:
        # Single-token prediction mode:
        with torch.no_grad():
            outputs = model(current_ids)
        logits = outputs.logits  # shape: (1, seq_length, vocab_size)
        next_token_logits = logits[0, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=0)
        top_prob, top_indices = torch.topk(probabilities, top_k)
        top_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_indices]
        
        print("Top {} predictions for the next token:".format(top_k))
        for token, prob in zip(top_tokens, top_prob):
            print(f"Token: '{token}' - Probability: {prob.item():.4f}")
    else:
        print("Iterative generation mode:")
        # Print the initial prompt.
        print("Initial token sequence:", " ".join(map(str, current_ids[0].tolist())))
        print("Decoded text:", tokenizer.decode(current_ids[0], skip_special_tokens=True))
        for i in range(num_predict):
            with torch.no_grad():
                outputs = model(current_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            probabilities = torch.softmax(next_token_logits, dim=0)
            # Greedy decoding: choose the token with the highest probability.
            next_token_id = torch.argmax(probabilities).unsqueeze(0).unsqueeze(0)
            # Append the predicted token to current_ids.
            current_ids = torch.cat((current_ids, next_token_id), dim=1)
            # Print the updated sequence.
            token_seq = current_ids[0].tolist()
            decoded = tokenizer.decode(current_ids[0], skip_special_tokens=True)
            #print(f"After {i+1} prediction(s):")
            #print("Token sequence:", " ".join(map(str, token_seq)))
            print("Decoded text:", decoded)



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
            print('Example: python token-demo.py predict_next_token "91 860 287 11579 3962"')
            sys.exit(1)
        token_string = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
        num_predict = int(sys.argv[4]) if len(sys.argv) >= 5 else 1
        predict_next_token(token_string, top_k, num_predict)

    else:
        print(f"Unknown command '{command}'.")
        print("Available commands: extract_samples, convert_text_to_utf8_byte_values, convert_text_to_utf8_bit_strings, convert_text_to_tokens, convert_tokens_to_text, predict_next_token")
        sys.exit(1)
