import re
from collections import Counter

def get_top_words(filename, top_n=100):
    # Create a dictionary to store word counts
    word_counts = Counter()
    
    # Read and process the file
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Convert to lowercase and remove punctuation
            # \W+ matches any non-word character
            words = re.split(r'\W+', line.lower())
            
            # Update counts for non-empty words
            for word in words:
                if word:  # Skip empty strings
                    word_counts[word] += 1
    
    # Get the top 100 most common words
    top_words = word_counts.most_common(top_n)
    
    # Print results
    for word, count in top_words:
        print(f"{word}: {count}")

# Usage
get_top_words('story.txt')