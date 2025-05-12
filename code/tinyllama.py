from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import torch
import math
from nltk.translate.bleu_score import sentence_bleu
import evaluate

# Load the Amazon Product Description dataset
dataset = load_dataset("Ateeqq/Amazon-Product-Description")

# Load the pre-trained TinyLlama-Chat model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")  # FP16 + Auto device (GPU/CPU)

# Function to generate a custom product description
def generate_custom_description(product_name, product_description, persona):
    # Define the instruction for the model
    input_text = f"""<|user|>
Generate a customized product description for the following persona without asking any further questions or making suggestions:
Persona: {persona}
Product Name: {product_name}
Product Description: {product_description}
Customized Product Description (only text, no questions, no suggestions):
<|assistant|>"""

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate the output
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process: Remove the input prompt part if necessary
    if "<|assistant|>" in generated_text:
        generated_text = generated_text.split("<|assistant|>")[-1].strip()

    if not generated_text.strip():
        generated_text = "Unable to generate a description for this product."


    return generated_text

# Function to calculate Perplexity
def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    log_likelihood = outputs.loss.item()
    perplexity = math.exp(log_likelihood)
    return perplexity

# Function to calculate BLEU score
def calculate_bleu(reference, generated):
    reference_tokens = [ref.split() for ref in reference]
    generated_tokens = generated.split()
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    return bleu_score

# Load ROUGE metric
rouge = evaluate.load("rouge")

# List of potential personas
personas = [
    "tight budget, student, looking for durability",
    "luxury seeker, prefers premium quality and brand",
    "eco-conscious, prefers sustainable and eco-friendly products",
    "tech enthusiast, loves innovative features",
    "fashion-conscious, looking for trendy and stylish products",
    "athlete, needs performance and durability"
]

# Loop through the dataset and generate descriptions
for i in range(3):  # For example, let's generate descriptions for the first 3 products in the dataset
    product_name = dataset['train'][i]['TITLE']
    product_description = dataset['train'][i]['DESCRIPTION']
    
    # Randomly select a persona for the current product
    selected_persona = random.choice(personas)
    
    # Generate the custom product description based on the persona
    custom_description = generate_custom_description(product_name, product_description, selected_persona)
    perplexity = calculate_perplexity(custom_description)

    bleu_score = calculate_bleu([product_description], custom_description)
    # rouge_score = rouge.compute(predictions=[custom_description], references=[product_description])
    
    # Print the results
    print(f"Product: {product_name}")
    print(f"Persona: {selected_persona}")
    print(f"Original Product Description: {product_description[:250]}...")  # Show part of the description
    print(f"Customized Product Description: {custom_description}\n\n")
    
    # Evaluation Results
    print(f"Perplexity of Generated Description: {perplexity}")
    print(f"BLEU Score: {bleu_score}")
    # print(f"ROUGE Score: {rouge_score}\n\n")
