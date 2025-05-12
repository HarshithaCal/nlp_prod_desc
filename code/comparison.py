import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import json
import logging
import csv

# Set up logging - for easier debugging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_inputs(title, description, persona):
    """Validate input data for generation."""
    if pd.isna(title) or pd.isna(description) or pd.isna(persona):
        logger.warning("Invalid input data: NaN values detected")
        return False
    if not isinstance(title, str) or not isinstance(description, str) or not isinstance(persona, str):
        logger.warning("Invalid input data: Non-string values detected")
        return False
    if not title.strip() or not description.strip() or not persona.strip():
        logger.warning("Invalid input data: Empty strings detected")
        return False
    return True

def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer from the specified path."""
    try:
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Successfully loaded model and tokenizer from {model_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None, None

def generate_description(model, tokenizer, title, description, persona, max_length=512):
    """Generate custom description using the model."""
    if not validate_inputs(title, description, persona):
        logger.warning("Skipping generation due to invalid inputs")
        return "Invalid input data"
        
    try:
        prompt = f"Title: {title}\nDescription: {description}\nPersona: {persona}\nCustom Description:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        
        # Create attention mask
        attention_mask = torch.ones_like(inputs["input_ids"])
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Custom Description:" in generated_text:
            generated_text = generated_text.split("Custom Description:")[-1].strip()
        
        if not generated_text:
            logger.warning("Generated empty text")
            return "Generation failed - empty output"
            
        return generated_text
    except Exception as e:
        logger.error(f"Error generating description: {str(e)}")
        return "Generation failed - error occurred"

def calculate_bleu(reference, candidate):
    """Calculate BLEU score between reference and candidate texts."""
    try:
        reference = str(reference) if reference is not None else ""
        candidate = str(candidate) if candidate is not None else ""
        
        if not reference.strip() or not candidate.strip():
            logger.warning("Empty reference or candidate text for BLEU calculation")
            return 0.0
            
        smoothie = SmoothingFunction().method1
        reference = reference.split()
        candidate = candidate.split()
        score = sentence_bleu([reference], candidate, smoothing_function=smoothie)
        return score
    except Exception as e:
        logger.error(f"Error calculating BLEU score: {str(e)}")
        return 0.0

def calculate_perplexity(model, tokenizer, text):
    """Calculate perplexity for given text using the model."""
    try:
        text = str(text) if text is not None else ""
        
        if not text.strip():
            logger.warning("Empty text for perplexity calculation")
            return float('inf')
            
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        perplexity = torch.exp(loss).item()
        
        if np.isnan(perplexity) or np.isinf(perplexity):
            logger.warning(f"Invalid perplexity value: {perplexity}")
            return float('inf')
            
        return perplexity
    except Exception as e:
        logger.error(f"Error calculating perplexity: {str(e)}")
        return float('inf')

def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts using SentenceBERT."""
    try:
        text1 = str(text1) if text1 is not None else ""
        text2 = str(text2) if text2 is not None else ""
        
        if not text1.strip() or not text2.strip():
            logger.warning("Empty texts for similarity calculation")
            return 0.0
            
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embedding1 = model.encode([text1])
        embedding2 = model.encode([text2])
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        if np.isnan(similarity):
            logger.warning("Invalid similarity value: NaN")
            return 0.0
            
        return similarity
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0.0

def read_custom_csv(file_path):
    """Read CSV file with custom handling for complex format."""
    data = []
    with open(file_path, 'r', encoding='latin-1') as f:
        # Skip header
        header = next(f)
        
        # Read file line by line
        current_row = []
        in_quotes = False
        current_field = ""
        
        for line in f:
            for char in line:
                if char == '"':
                    if in_quotes and current_field.endswith('"'):
                        current_field += char
                    else:
                        in_quotes = not in_quotes
                    current_field += char
                elif char == ',' and not in_quotes:
                    # End of field
                    current_row.append(current_field.strip('"'))
                    current_field = ""
                else:
                    current_field += char
            
            # If we're not in quotes, this is the end of a row
            if not in_quotes:
                if current_field:
                    current_row.append(current_field.strip('"'))
                if len(current_row) == 4:  # We expect 4 columns
                    data.append(current_row)
                current_row = []
                current_field = ""
            else:
                # We're in quotes, so this line continues the current field
                current_field += "\n"
    
    # Convert to DataFrame
    return pd.DataFrame(data, columns=['TITLE', 'DESCRIPTION', 'PERSONA', 'CUSTOM DESCRIPTION'])

def evaluate_models(data_path, model_paths):
    """Evaluate all models and compare their performance."""
    try:
        logger.info(f"Reading data from {data_path}")
        # Read CSV with custom parameters to handle the complex format
        df = pd.read_csv(
            data_path,
            quoting=csv.QUOTE_ALL,  # Quote all fields
            doublequote=True,  # Allow double quotes
            escapechar='\\',  # Use backslash as escape character
            encoding='latin-1'  # Use latin-1 encoding for special characters
        )
        
        # Validate data
        if df.empty:
            logger.error("Empty dataframe loaded")
            return {}
            
        # Select only rows 102-113 which have complete data
        df = df.iloc[120:145]  # End index is exclusive
        logger.info(f"Processing rows 120-145 ({len(df)} rows)")
        
        # Convert all text columns to string type and clean them
        text_columns = ['TITLE', 'DESCRIPTION', 'PERSONA', 'CUSTOM DESCRIPTION']
        
        if len(df) == 0:
            logger.error("No valid rows after filtering")
            return {}
            
        results = {}
        model_outputs = {
            'Title': [], #We did not use this column in the generation in 2nd version of the code.
            'Description': [],
            'Persona': [],
            'Expected_Output': [],
        }
        
        # Initialize model output columns
        for model_path in model_paths:
            model_name = os.path.basename(model_path)
            model_outputs[f'Generated_{model_name}'] = []
        
        for model_path in model_paths:
            model_name = os.path.basename(model_path)
            logger.info(f"\nEvaluating model: {model_name}")
            
            model, tokenizer = load_model_and_tokenizer(model_path)
            if model is None or tokenizer is None:
                logger.error(f"Skipping evaluation for {model_name} due to loading failure")
                continue
            
            bleu_scores = []
            perplexities = []
            similarities = []
            
            # Only populate the common columns once
            if len(model_outputs['Title']) < len(df):
                model_outputs['Title'].extend(df['TITLE'].tolist())
                model_outputs['Description'].extend(df['DESCRIPTION'].tolist())
                model_outputs['Persona'].extend(df['PERSONA'].tolist())
                model_outputs['Expected_Output'].extend(df['CUSTOM DESCRIPTION'].tolist())
            
            total_items = len(df) 
            for i, (idx, row) in enumerate(df.iterrows()):
                logger.info(f"Processing item {i + 1}/{total_items}")
                reference = row['CUSTOM DESCRIPTION']
                
                # Generate description using the model
                candidate = generate_description(
                    model, 
                    tokenizer, 
                    row['TITLE'], 
                    row['DESCRIPTION'], 
                    row['PERSONA']
                )
                
                # Store generated output
                model_outputs[f'Generated_{model_name}'].append(candidate)
                # Calculate metrics
                bleu = calculate_bleu(reference, candidate)
                perplexity = calculate_perplexity(model, tokenizer, reference)
                perplexities.append(float(perplexity))
                similarity = calculate_similarity(reference, candidate)
                bleu_scores.append(float(bleu))
                similarities.append(float(similarity))
            
            # Calculate average scores and convert to Python floats
            results[model_name] = {
                'average_bleu': float(np.mean(bleu_scores)),
                'average_perplexity': float(np.mean(perplexities)),
                'average_similarity': float(np.mean(similarities))
            }
            logger.info(f"Completed evaluation for {model_name}")
        
        # Save results
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        logger.info("Saved evaluation results to evaluation_results.json")
        
        # Create and save comparison DataFrame
        comparison_df = pd.DataFrame(model_outputs)
        comparison_df.to_csv('model_outputs_comparison.csv', index=False)
        logger.info("Saved model outputs comparison to model_outputs_comparison.csv")
        
        return results
    except Exception as e:
        logger.error(f"Error in evaluate_models: {str(e)}")
        return {}

def main():
    try:
        # Specify paths to your models and test data
        model_paths = [
            'models/tinyllama_product_customizer_final',
            'models/fine_tuned_distilgpt2',
            # 'models/gemma_3_4b_it'
        ]
        
        data_path = 'data/first_300_finetune_updated.csv'
        
        # Run evaluation
        results = evaluate_models(data_path, model_paths)
        
        # Print results
        print("\nEvaluation Results:")
        print("==================")
        for model_name, scores in results.items():
            print(f"\nModel: {model_name}")
            print(f"Average BLEU Score: {scores['average_bleu']:.4f}")
            print(f"Average Perplexity: {scores['average_perplexity']:.4f}")
            print(f"Average Similarity: {scores['average_similarity']:.4f}")
        
        print("\nOutputs comparison saved to 'model_outputs_comparison.csv'")
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 