from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import torch
import math
import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate 
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
# from finetune_tinyllama import calculate_perplexity, similarity_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
login(token="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")
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2') #,cache_folder="/ocean/projects/mth240012p/dwarache/data/")
def calculate_similarity(text1, text2, sentence_transformer):
    """Calculate cosine similarity between two texts using SentenceBERT."""
    try:
        if text1 is not None:
            text1 = str(text1)
        else:
            text1 = ""
        if text2 is not None:
            text2 = str(text2)
        else:
            text2 = ""
        
        if not text1.strip() or not text2.strip():
            return 0.0

        #Encode the texts to embeddings for similarity calculation  
        embedding1 = sentence_transformer.encode([text1])
        embedding2 = sentence_transformer.encode([text2])
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        if np.isnan(similarity):
            return 0.0

        return similarity
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0

#  Global variables to store metrics
train_perplexities = []
val_perplexities = []
train_similarities = []
val_similarities = []
steps = []

def add_metrics(step, train_perplexity, val_perplexity, train_similarity, val_similarity):
    """Add metrics for a training step"""
    steps.append(step)
    train_perplexities.append(train_perplexity)
    val_perplexities.append(val_perplexity)
    train_similarities.append(train_similarity)
    val_similarities.append(val_similarity)

def plot_metrics():
    """Plot and save training metrics"""
    # Create a timestamp for the plot filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create DataFrame for plotting
    metrics_df = pd.DataFrame({
        'Steps': steps,
        'Train Perplexity': train_perplexities,
        'Validation Perplexity': val_perplexities,
        'Train Similarity': train_similarities,
        'Validation Similarity': val_similarities
    })
    
    # Plot perplexity
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=metrics_df, x='Steps', y='Train Perplexity', label='Train', marker='o')
    sns.lineplot(data=metrics_df, x='Steps', y='Validation Perplexity', label='Validation', marker='s')
    plt.title('Training and Validation Perplexity Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'output/plots/perplexity_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot cosine similarity
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=metrics_df, x='Steps', y='Train Similarity', label='Train', marker='o')
    sns.lineplot(data=metrics_df, x='Steps', y='Validation Similarity', label='Validation', marker='s')
    plt.title('Training and Validation Cosine Similarity Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'output/plots/similarity_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics data
    metrics_df.to_csv(f'output/plots/metrics_{timestamp}.csv', index=False)
# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Create output directory for plots
os.makedirs('../output/plots', exist_ok=True)

# Load only the test data
print("Loading test data...")
df = pd.read_csv("data/cleaned_data.csv")
test_data = df[120:144]  # Only using test data (rows 120-143)

# Load the pre-trained Gemma model and tokenizer
# custom_dir = "/ocean/projects/mth240012p/dwarache/data/"
model_name = "google/gemma-3-4b-it" 
tokenizer = AutoTokenizer.from_pretrained(model_name)#, custom_dir=custom_dir)
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it")#, cache_dir="/ocean/projects/mth240012p/dwarache/data/")
print("Model and tokenizer loaded successfully.")

# Function to generate a custom product description
def generate_custom_description(product_description, persona):
    # Define the instruction for the model
    input_text = f"""Task: Create a customized product description for the following persona.
                    Persona: {persona}
                    Product: {product_description}
                    Requirements:
                    - Maintain the original product type
                    - Focus on features relevant to the persona
                    - Keep the description concise and engaging
                    Customized description:"""

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate the output (customized product description)
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,  # Lower temperature for more focused output
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,  # Add repetition penalty
        do_sample=True
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the customized description part
    try:
        # Find the start of the customized description
        start_idx = generated_text.find("Customized Product Description")
        if start_idx != -1:
            # Find the actual start of the description after the colon
            start_idx = generated_text.find(":", start_idx) + 1
            if start_idx > 0:
                # Get everything after the colon and clean it up
                description = generated_text[start_idx:].strip()
                # Remove any trailing text after the description
                end_idx = description.find("\n")
                if end_idx != -1:
                    description = description[:end_idx].strip()
                return description
    except Exception as e:
        print(f"Error extracting description: {str(e)}")
    
    # If extraction fails, return the full generated text
    return generated_text
def calculate_perplexity(model, tokenizer, text):
    """Calculate perplexity for given text using the model."""
    try:
        text = str(text) if text is not None else ""
        
        if not text.strip():
            return float('inf')
            
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        perplexity = torch.exp(loss).item()
        
        if np.isnan(perplexity) or np.isinf(perplexity):
            return float('inf')
            
        return perplexity
    except Exception as e:
        return float('inf')
# Initialize lists to store metrics
perplexities = []
bleu_scores = []
similarity_scores = []

# Test the model on test data
print("\nTesting model on test data...")
for i, row in test_data.iterrows():
    try:
        product_description = row['DESCRIPTION']
        persona = row['PERSONA']
        expected_description = row['CUSTOM DESCRIPTION']
        
        print(f"\nProcessing Test Sample {i+1}:")
        print(f"Persona: {persona}")
        print(f"Original Description: {product_description[:250]}...")
        
        # Generate custom description
        custom_description = generate_custom_description(product_description, persona)
        print("\nGenerated Description:")
        print("-" * 50)
        print(custom_description)
        print("-" * 50)
        
        # Calculate metrics
        perplexity = calculate_perplexity(model, tokenizer, custom_description)
        bleu_score = sentence_bleu([word_tokenize(expected_description)], 
                                 word_tokenize(custom_description),
                                 smoothing_function=SmoothingFunction().method1)
        similarity = calculate_similarity(custom_description, expected_description, sentence_transformer)
        
        # Store metrics
        perplexities.append(perplexity)
        bleu_scores.append(bleu_score)
        similarity_scores.append(similarity)
        
        # Print metrics
        print(f"\nExpected Description:")
        print("-" * 50)
        print(expected_description)
        print("-" * 50)
        print(f"\nMetrics:")
        print(f"Perplexity: {perplexity:.4f}")
        print(f"BLEU Score: {bleu_score:.4f}")
        print(f"Similarity Score: {similarity:.4f}")
        print("=" * 100)
            
    except Exception as e:
        print(f"Error processing sample {i+1}: {str(e)}")
        continue

# Calculate and print average metrics
if perplexities:
    print("\nAverage Metrics:")
    print(f"Average Perplexity: {np.mean(perplexities):.4f}")
    print(f"Average BLEU Score: {np.mean(bleu_scores):.4f}")
    print(f"Average Similarity Score: {np.mean(similarity_scores):.4f}")
    
    # Create plots for each metric
    plt.figure(figsize=(15, 5))
    
    # Perplexity plot
    plt.subplot(1, 3, 1)
    plt.plot(perplexities, 'b-o')
    plt.title('Perplexity Distribution')
    plt.xlabel('Sample Index')
    plt.ylabel('Perplexity')
    plt.grid(True)
    
    # BLEU Score plot
    plt.subplot(1, 3, 2)
    plt.plot(bleu_scores, 'g-o')
    plt.title('BLEU Score Distribution')
    plt.xlabel('Sample Index')
    plt.ylabel('BLEU Score')
    plt.grid(True)
    
    # Similarity Score plot
    plt.subplot(1, 3, 3)
    plt.plot(similarity_scores, 'r-o')
    plt.title('Similarity Score Distribution')
    plt.xlabel('Sample Index')
    plt.ylabel('Similarity Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../output/plots/metrics_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a combined metrics plot
    plt.figure(figsize=(10, 6))
    plt.plot(perplexities, 'b-o', label='Perplexity')
    plt.plot(bleu_scores, 'g-o', label='BLEU Score')
    plt.plot(similarity_scores, 'r-o', label='Similarity Score')
    plt.title('Metrics Distribution')
    plt.xlabel('Sample Index')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../output/plots/combined_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics data to CSV
    metrics_df = pd.DataFrame({
        'Perplexity': perplexities,
        'BLEU_Score': bleu_scores,
        'Similarity_Score': similarity_scores
    })
    metrics_df.to_csv('../output/plots/metrics_data.csv', index=False)
    
else:
    print("\nNo valid metrics were calculated.")
