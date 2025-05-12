import pandas as pd

df = pd.read_csv(r'C:\Users\DELL\Desktop\sem2\INFO259_NLP\Project\data\first_300_train.csv')

df_duplicated = pd.DataFrame(df.values.repeat(6, axis=0), columns=df.columns)

df_duplicated.to_csv(r'C:\Users\DELL\Desktop\sem2\INFO259_NLP\Project\data\first_300_finetune.csv', index=False)

# Load the CSV file
df = pd.read_csv(r'C:\Users\DELL\Desktop\sem2\INFO259_NLP\Project\data\first_300_finetune.csv')

# Define the sequence to repeat in the 'PERSONA' column
persona_sequence = [
    "tight budget, student, looking for durability", 
    "luxury seeker, prefers premium quality and brand", 
    "eco-conscious, prefers sustainable and eco-friendly products", 
    "tech enthusiast, loves innovative features", 
    "fashion-conscious, looking for trendy and stylish products", 
    "athlete, needs performance and durability"
]

# Repeat the sequence to match the length of the 'PERSONA' column
num_repeats = len(df) // len(persona_sequence) + 1  # Ensure enough repetitions
persona_repeated = (persona_sequence * num_repeats)[:len(df)]  # Trim to exact length

# Replace the 'PERSONA' column with the repeated sequence
df['PERSONA'] = persona_repeated

# Save the modified dataframe to a new CSV file
df.to_csv(r'C:\Users\DELL\Desktop\sem2\INFO259_NLP\Project\data\first_300_finetune_updated.csv', index=False)

print("PERSONA column updated and saved successfully!")

print("Rows duplicated successfully!")
