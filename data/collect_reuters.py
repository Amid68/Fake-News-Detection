import feedparser
import pandas as pd
import os
import time

def collect_reuters_titles():
    print("Collecting Reuters titles via Google News...")
    
    # Google News RSS feeds specifically for Reuters content
    rss_feeds = [
        'https://news.google.com/rss/search?q=site:reuters.com+when:7d',  # All Reuters content from last 7 days
        'https://news.google.com/rss/search?q=site:reuters.com+politics+when:30d',  # Political news
        'https://news.google.com/rss/search?q=site:reuters.com+world+when:30d',     # World news
        'https://news.google.com/rss/search?q=site:reuters.com+government+when:30d' # Government news
    ]
    
    # File path for dataset
    file_path = 'manual_real.csv'
    
    # Load existing dataset if it exists
    try:
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            print(f"Loaded existing dataset with {len(existing_df)} articles")
            existing_titles = set(existing_df['title']) if 'title' in existing_df.columns else set()
        else:
            existing_df = pd.DataFrame(columns=['title', 'text'])
            existing_titles = set()
            print("No existing dataset found. Creating new dataset.")
    except Exception as e:
        print(f"Error loading existing dataset: {e}")
        existing_df = pd.DataFrame(columns=['title', 'text'])
        existing_titles = set()
    
    # Collect new articles (titles only)
    new_articles = []
    processed_titles = set()
    
    # Process each RSS feed
    for feed_url in rss_feeds:
        print(f"\nProcessing feed: {feed_url}")
        
        # Parse the RSS feed
        feed = feedparser.parse(feed_url)
        print(f"Found {len(feed.entries)} entries")
        
        for entry in feed.entries:
            # Extract article title (clean up Reuters suffix if present)
            title = entry.title.replace(' - Reuters', '').strip()
            
            # Skip if already in our dataset or processed in this run
            if title in existing_titles or title in processed_titles:
                continue
                
            # Add to our collection - use title as both title and text
            new_articles.append({
                'title': title,
                'text': title  # Just use the title as text
            })
            
            processed_titles.add(title)
            print(f"Added: {title[:60]}...")
    
    print(f"\nTotal new titles collected: {len(new_articles)}")
    
    # If we found new articles, add them to the dataset
    if new_articles:
        # Create DataFrame for new articles
        new_df = pd.DataFrame(new_articles)
        
        # Combine with existing dataset
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Save to CSV
        combined_df.to_csv(file_path, index=False)
        
        print(f"Success! Added {len(new_df)} new article titles to the dataset.")
        print(f"Total articles in dataset: {len(combined_df)}")
    else:
        print(f"\nNo new article titles found to add to the dataset.")

if __name__ == "__main__":
    collect_reuters_titles()
