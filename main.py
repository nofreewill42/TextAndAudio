
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer,
)

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_compute_type=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=True,
        # load_in_8bit=True,
    )

    # Load model
    tokenizer = LlamaTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=False,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        trust_remote_code=False,
        use_safetensors=True,
        quantization_config=bnb_config,
    )

    return tokenizer, model

tokenizer, model = load_llm()

# Quantize the model to GPU


# Define the sampler function
@torch.no_grad
def sampler(input_ids):
    outputs = []

    for _ in range(50):
        res = model(input_ids=input_ids)
        # res.logits shape is (batch, seq_len, logits)
        # sample using multinomial using the last logits 
        sampled = torch.multinomial(res.logits[:,-1,:].softmax(dim=-1), 1)
        # repeatedly concat the `sampled` to the `input_ids` for next sampling
        input_ids = torch.cat((input_ids, sampled), dim=-1)

    return input_ids

# Define the prompt
prompt = "The quick black gazelle"

# Tokenize the prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

# Generate text
output = sampler(input_ids)

# Decode the output
output_text = tokenizer.decode(output[0])

print(output_text)