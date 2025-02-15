# Token Demo

This repository contains a demonstration script for working with text tokenization and next-token prediction using GPT2. The script provides several utility functions that:

- Extract a small sample of text data from the FineWeb dataset.
- Convert the text into various representations:
  - UTF-8 bytes (numbers)
  - 8-bit binary strings (bits)
  - GPT2 token IDs
- Convert a string of GPT2 token IDs back into text (printed to the console).
- Predict the next token from a given sequence of token IDs and display the top choices with their probability distribution.

## Requirements

- Python 3.7 or higher

You can install the required packages using:

```bash
pip install transformers torch datasets
```

## Running the code

```bash
python token-demo.py extract_samples
python token-demo.py convert_text_to_utf8_bit_strings
python token-demo.py convert_text_to_utf8_byte_values
python token-demo.py convert_text_to_tokens
python token-demo.py convert_tokens_to_text "91 7680 278 14206 2947 3574 25 1338 9437" 
python token-demo.py predict_next_token "91 7680 278 14206 2947 3574 25 1338 9437"
python token-demo.py predict_next_token "91 7680 278 14206 2947 3574 25 1338 9437" 1 20
python token-demo.py predict_next_token "Whenever there is summer outside" 1 20
```