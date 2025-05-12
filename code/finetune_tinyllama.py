import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline
from datasets import Dataset
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# Create output directory for plots
os.makedirs('output/plots', exist_ok=True)

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

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

# Global variables to store metrics
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

# Load the CSV file with the correct encoding
print("Loading data...")
df = pd.read_csv(r'data\cleaned_data.csv')
print(df.head())

# Split the data 
train_data = df[:120]
test_data = df[120:144]

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

# Function to format the data into prompts
def format_data(data):
    formatted_data = []
    for _, row in data.iterrows():
        product_description = row['DESCRIPTION']
        persona = row['PERSONA']
        custom_description = row['CUSTOM DESCRIPTION']
        
        # More structured prompt
        input_prompt = f"""Task: Create a customized product description for the following persona.
                            Persona: {persona}
                            Product: {product_description}
                            Requirements:
                            - Maintain the original product type
                            - Focus on features relevant to the persona
                            - Keep the description concise and engaging
                            Customized description:"""
                                    
        formatted_data.append({
            'text': f"{input_prompt} {custom_description}"
        })
    return formatted_data

# Format the train and test data
print("Formatting data...")
train_data = format_data(train_data)
test_data = format_data(test_data)

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# Load TinyLlama model and tokenizer
print("Loading TinyLlama model...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configure LoRA
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=32,  # rank - number of trainable parameters in the LoRA.
    lora_alpha=64,  # alpha scaling - more the value more the weight of the LoRA.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # target attention modules
    lora_dropout=0.05,
    bias="none", 
    task_type=TaskType.CAUSAL_LM #Causual LM for text generation.
)

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# Tokenize the dataset
def tokenize_function(examples):
    "For batch processing and to format compatible with the HF dataset format"
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

print("Tokenizing dataset...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True) #map() processes the dataset in memory-efficient manner
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama_product_customizer",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    optim="adamw_torch", #adamw_torch is a variant of the Adam optimizer that uses weight decay and is optimized to work with LLMs
    max_grad_norm=1.0,
    seed=42
)

# Create data collator for efficient data preparation and processing. This is optional but recommended.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

train_val_split = tokenized_train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Load sentence transformer for cosine similarity.
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    # Calculate perplexity and similarity
    batch_train_perplexities = []
    batch_val_perplexities = []
    batch_train_similarities = []
    batch_val_similarities = []
    
    for pred, label in zip(predictions, labels):
        try:
            # Convert to text
            pred_text = tokenizer.decode(pred, skip_special_tokens=True)
            label_text = tokenizer.decode(label, skip_special_tokens=True)
            
            # Calculate perplexity
            train_perplexity = calculate_perplexity(model, tokenizer, pred_text)
            val_perplexity = calculate_perplexity(model, tokenizer, label_text)
            
            # Calculate similarity using the existing sentence_transformer
            train_similarity = calculate_similarity(pred_text, label_text, sentence_transformer)
            val_similarity = calculate_similarity(label_text, pred_text, sentence_transformer)
            
            batch_train_perplexities.append(train_perplexity)
            batch_val_perplexities.append(val_perplexity)
            batch_train_similarities.append(train_similarity)
            batch_val_similarities.append(val_similarity)
            
        except Exception as e:
            print(f"Error in compute_metrics: {str(e)}")
            continue
    
    # Calculate averages
    avg_train_perplexity = np.mean(batch_train_perplexities) if batch_train_perplexities else float('inf')
    avg_val_perplexity = np.mean(batch_val_perplexities) if batch_val_perplexities else float('inf')
    avg_train_similarity = np.mean(batch_train_similarities) if batch_train_similarities else 0.0
    avg_val_similarity = np.mean(batch_val_similarities) if batch_val_similarities else 0.0
    
    # Store metrics
    add_metrics(
        trainer.state.global_step,
        avg_train_perplexity,
        avg_val_perplexity,
        avg_train_similarity,
        avg_val_similarity
    )
    
    return {
        "train_perplexity": avg_train_perplexity,
        "val_perplexity": avg_val_perplexity,
        "train_similarity": avg_train_similarity,
        "val_similarity": avg_val_similarity
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# Save the fine-tuned model - important else we need to retrain the model from scratch everytime.
print("Saving fine-tuned model...")
trainer.save_model("./tinyllama_product_customizer_final")

print("Loading fine-tuned model for testing...")
model = AutoModelForCausalLM.from_pretrained("./tinyllama_product_customizer_final")
tokenizer = AutoTokenizer.from_pretrained("./tinyllama_product_customizer_final")

# Create a text generation pipeline
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    do_sample=True
)

# Test with samples from the test set
print("\nTesting model on test samples...")
similarities = []
perplexities = []

for i, test_sample in enumerate(tqdm(test_data, desc="Generating responses")):
    try:
        input_prompt = test_sample['text'].split("Customized description:")[0] + "Customized description:"
        
        # Generate response using the fine-tuned model with optimized parameters
        response = generator(
            input_prompt,
            max_length=512,
            truncation=True,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
        
        # Get the generated text
        generated_text = response[0]['generated_text']
        print("\nFull Generated Text:")
        print(generated_text)
        
        # Extract only the customized product description part
        start_idx = generated_text.find("Customized description:")
        if start_idx != -1:
            generated_description = generated_text[start_idx + len("Customized description:"):].strip()
            # Remove any trailing text after the description
            end_idx = generated_description.find("\n")
            if end_idx != -1:
                generated_description = generated_description[:end_idx].strip()
        else:
            generated_description = generated_text
        
        # Get the expected description from the test sample
        expected_text = test_sample['text'].split("Customized description:")[1].strip()
        
        # Calculate cosine similarity
        embeddings = sentence_transformer.encode([generated_description, expected_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        similarities.append(similarity)
        
        # Calculate perplexity
        perplexity = calculate_perplexity(model, tokenizer, generated_description)
        perplexities.append(perplexity)
        
        # Print the results
        print(f"\nTest Sample {i+1}:")
        print("Generated Description:", generated_description)
        print("\nExpected Description:", expected_text)
        print(f"\nCosine Similarity: {similarity:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
        print("-" * 100)
        
    except Exception as e:
        print(f"Error processing sample {i+1}: {str(e)}")
        continue

# Print average metrics
if similarities:
    print("\nAverage Metrics:")
    print(f"Cosine Similarity: {np.mean(similarities):.4f}")
    print(f"Perplexity: {np.mean(perplexities):.4f}")
else:
    print("\nNo valid metrics were calculated.")

# After training is complete, plot the metrics
print("\nGenerating training metrics plots...")
plot_metrics()
print("Training metrics plots saved.") 