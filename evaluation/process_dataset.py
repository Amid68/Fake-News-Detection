# process_dataset.py
import pandas as pd

# Load your dataset (adjust path as needed)
input_file = './datasets/FakeNewsNet.csv'
output_file = './datasets/simplified_FakeNewsNet.csv'

print(f"Processing dataset from {input_file}...")

# Load the dataset
df = pd.read_csv(input_file)
print(f"Loaded dataset with {len(df)} articles and columns: {df.columns.tolist()}")

# Keep only title and real columns (assuming 'real' is your label column)
df = df[['title', 'real']]

# Rename 'real' to 'label' to match your evaluation code
df = df.rename(columns={'real': 'label'})

# Save the simplified dataset
df.to_csv(output_file, index=False)
print(f"Saved simplified dataset with {len(df)} articles to {output_file}")
