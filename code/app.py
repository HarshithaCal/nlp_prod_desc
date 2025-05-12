from flask import Flask, request, jsonify
from flask_cors import CORS
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import torch
import math
from nltk.translate.bleu_score import sentence_bleu
import evaluate
from finetune_tinyllama import calculate_perplexity

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the Amazon Product Description dataset for product description.
dataset = load_dataset("Ateeqq/Amazon-Product-Description")

# Load the pre-trained Gemma model and tokenizer
model_name = "./tinyllama_product_customizer_final"   # change to gemma_3_4b_it for gemma_3_4b_it
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

personas = [
    "tight budget, student, looking for durability",
    "luxury seeker, prefers premium quality and brand",
    "eco-conscious, prefers sustainable and eco-friendly products",
    "tech enthusiast, loves innovative features",
    "fashion-conscious, looking for trendy and stylish products",
    "athlete, needs performance and durability"
]


def generate_custom_description(product_name, product_description, persona):
    # Define the instruction for the model
    # input_text = f"""Generate a customized product description for the following persona without asking any further questions or making suggestions:
    #             Persona: {persona}
    #             Product Name: {product_name}
    #             Product Description: {product_description}
    #             Customized Product Description (only text, no questions, no suggestions):"""

    input_text = f"""Task: Create a customized product description for the following persona.
                    Persona: {persona}
                    Product: {product_description}
                    Requirements:
                    - Maintain the original product type
                    - Focus on features relevant to the persona
                    - Keep the description concise and engaging
                    Customized description:"""   # this prompt works way better.


    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    # customized product description generation.
    outputs = model.generate(**inputs, max_length=512, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the customized description part
    if "Customized Product Description" in generated_text:
        parts = generated_text.split("Customized Product Description")
        if len(parts) > 1:
            custom_part = parts[1].strip()
            return custom_part
    
    # Return the full generated text if we can't extract just the description part
    return generated_text

# Function to calculate Perplexity
def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
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

# # Load ROUGE metric
rouge = evaluate.load("rouge")

# Define an API route to generate descriptions
@app.route('/generate_description', methods=['POST'])

def generate_description():
    # Extract JSON data from the request body
    data = request.get_json()

    # Extract necessary fields
    product_name = data.get('product_name', '')
    product_description = data.get('product_description', '')
    persona = data.get('persona', '')

    if not product_name or not product_description or not persona:
        return jsonify({'error': 'Missing required fields'}), 400

    # Generate the custom description
    custom_description = generate_custom_description(product_name, product_description, persona)

    # Evaluate perplexity, BLEU, and ROUGE scores
    perplexity = calculate_perplexity(custom_description)
    bleu_score = calculate_bleu([product_description], custom_description)
    # rouge_score = rouge.compute(predictions=[custom_description], references=[product_description])

    # Convert BLEU scores to serializable format
    serializable_rouge = {}
    for key, value in bleu_score.items():
        serializable_rouge[key] = float(value)

    # Return the generated description and evaluation metrics
    return jsonify({
        'product_name': product_name,
        'persona': persona,
        'custom_description': custom_description,
        'perplexity': perplexity,
        'bleu_score': bleu_score,
        # 'rouge_score': serializable_rouge
    })

if __name__ == '__main__':
    app.run(debug=True)
