from datasets import load_dataset
import pandas as pd

# Load the original dataset 
print("Loading original dataset...")
dataset = load_dataset("Ateeqq/Amazon-Product-Description")
data = dataset['train']
df = data.to_pandas()
df_first_300 = df.head(300) # 300*6 = 1800 samples. here, 6 is number of personas.
df_first_300.to_csv("first_300_train.csv", index=False)

# Take the next 100 rows - validation data
# df_next_300 = df.iloc[300:400]
# df_next_300.to_csv("next_100_val.csv", index=False)

print("Saved !!!!!")
