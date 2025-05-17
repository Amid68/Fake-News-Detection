from datasets import load_dataset

# Load the politifact subset (political news)
dataset = load_dataset("fakenewsnet", "politifact")
print(f"Dataset loaded with {len(dataset['train'])} articles")

# View structure
print(dataset['train'][0])

# Extract fake news articles
fake_news = dataset["train"].filter(lambda example: example["label"] == 1)
print(f"Extracted {len(fake_news)} fake news articles")

# Convert to pandas for easier manipulation
fake_news_df = fake_news.to_pandas()
fake_news_df = fake_news_df[['news_url', 'title', 'text', 'label']]

# Save to CSV for your evaluation pipeline
fake_news_df.to_csv('./datasets/fakenewsnet_fake.csv', index=False)
