import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from datasets import Dataset
from transformers import TrainingArguments, Trainer, pipeline
import torch
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Function to try different encodings - not needed as we corrected the corrupted csv file.
# def read_csv_with_encoding(file_path):
#     encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
#     for encoding in encodings:
#         try:
#             print(f"Trying encoding: {encoding}")
#             df = pd.read_csv(file_path, encoding=encoding)
#             print(f"Successfully read CSV with {encoding} encoding")
#             return df
#         except  e:
#             print(f"Failed with {encoding}: {str(e)}")
#             continue

# Load the CSV file with the correct encoding
df = pd.read_csv(r'data/cleaned_data.csv')

print(df.head())

train_data = df.iloc[:120]
test_data = df.iloc[120:144]

# Function to format the data into prompts
def format_data(data):
    formatted_data = []
    for i, row in data.iterrows():
        product_description = row['DESCRIPTION']
        persona = row['PERSONA']
        custom_description = row['CUSTOM DESCRIPTION']
        
        # Creating a shorter input prompt as we will run out of tokens.
        input_prompt = f"Persona: {persona}\nProduct: {product_description}\nCustomized description:"
        
        formatted_data.append({
            'input_prompt': input_prompt,
            'desired_product_description': custom_description
        })
    return formatted_data

# Format the train and test data
train_data = format_data(train_data)
test_data = format_data(test_data)

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# Load tokenizer and model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad token for DistilGPT2
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define LoRA configuration for the model
lora_config = LoraConfig(
    r=16,                        # Low-rank size
    lora_alpha=32,              # Scaling factor for low-rank matrices
    lora_dropout=0.1,           # Dropout rate for LoRA layers
    bias="all",                 # Apply LoRA to all biases
    task_type="CAUSAL_LM"       # Task type: causal language modeling
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Tokenization function
def tokenize_function(examples):
    # Tokenize the input prompt
    input_prompt = examples["input_prompt"]
    model_inputs = tokenizer(input_prompt, truncation=True, padding="max_length", max_length=512)
    
    # Tokenize the desired product description (output)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["desired_product_description"], truncation=True, padding="max_length", max_length=512)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization to the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",            # Directory to save model checkpoints
    num_train_epochs=3,                # Number of training epochs
    per_device_train_batch_size=4,     # Batch size per device during training
    save_strategy="epoch",             # Save checkpoints at the end of each epoch
    logging_dir="./logs",              # Directory for storing logs
    logging_steps=500,                 # Log training progress every 500 steps
    warmup_steps=500,                  # Warmup steps for learning rate scheduler
    weight_decay=0.01,                 # Weight decay to avoid overfitting
    fp16=True,                         # Enable mixed precision training if using GPU
    eval_strategy="epoch",             # Evaluate at the end of each epoch
    load_best_model_at_end=True,       # Load the best model at the end of training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                       # Model to train
    args=training_args,                # Training arguments
    train_dataset=tokenized_train_dataset,  # Training dataset
    eval_dataset=tokenized_test_dataset,   # Evaluation dataset
)

# Fine-tune the model with progress bar
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model and tokenizer
print("\nSaving fine-tuned model...")
model.save_pretrained("./models/fine_tuned_distilgpt2")
tokenizer.save_pretrained("./models/fine_tuned_distilgpt2")

# Load the fine-tuned model and tokenizer for testing
print("Loading fine-tuned model for testing...")
model = AutoModelForCausalLM.from_pretrained("./models/fine_tuned_distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("./models/fine_tuned_distilgpt2")

# Test the fine-tuned model (custom generation)
model.eval()

# Create a text generation pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Load sentence transformer for cosine similarity
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

# Test with samples from the test set
print("\nTesting model on test samples...")
similarities = []
for i, test_sample in enumerate(tqdm(test_data, desc="Generating responses")):
    input_prompt = test_sample['input_prompt']
    
    # Generate response using the fine-tuned model with adjusted parameters
    response = generator(input_prompt, 
                         max_length=512,          # Reduced from 1024
                         num_return_sequences=1,
                         truncation=True,
                         top_k=50,
                         top_p=0.9,              # Adjusted for more creativity
                         do_sample=True,
                         temperature=0.8,         # Increased for more variation
                         pad_token_id=tokenizer.eos_token_id,
                         no_repeat_ngram_size=4,  # Prevent repetition of 4-word phrases
                         repetition_penalty=1.5)   # Penalize repeated tokens
    
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
    
    # Calculate cosine similarity between generated and expected text
    expected_text = test_sample['desired_product_description']
    embeddings = sentence_transformer.encode([generated_description, expected_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    similarities.append(similarity)
    
    # Print the results
    print(f"\nTest Sample {i+1}:")
    print("Generated Description:", generated_description)
    print("\nExpected Description:", expected_text)
    print(f"\nCosine Similarity: {similarity:.4f}")
    print("-" * 100)

# Print average similarity
print(f"\nAverage Cosine Similarity across all test samples: {np.mean(similarities):.4f}")
