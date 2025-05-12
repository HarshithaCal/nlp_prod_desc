import pandas as pd
import numpy as np
import re

df = pd.read_csv(r"C:\Users\DELL\Desktop\sem2\INFO259_NLP\Project\data\first_300_finetune_updated.csv")

df.head()

# Remove HTML tags
df['DESCRIPTION'] = df['DESCRIPTION'].apply(lambda x: re.sub(r'<.*?>', '', x))

# Remove ® like text
df['DESCRIPTION'] = df['DESCRIPTION'].apply(lambda x: re.sub(r'®', '', x))
# Remove special characters
df['DESCRIPTION'] = df['DESCRIPTION'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

# Remove extra whitespace
df['DESCRIPTION'] = df['DESCRIPTION'].apply(lambda x: re.sub(r'\s+', ' ', x))
df['DESCRIPTION'] = df['DESCRIPTION'].apply(lambda x: x.strip())

# Remove non-ascii characters
df['DESCRIPTION'] = df['DESCRIPTION'].apply(lambda x: x.encode('ascii', 'ignore').decode())


df.to_csv("cleaned_data.csv", index=False)

df = pd.read_csv(r"C:\Users\DELL\Desktop\sem2\INFO259_NLP\Project\cleaned_data.csv")
print(df.head())
print(df.columns)



