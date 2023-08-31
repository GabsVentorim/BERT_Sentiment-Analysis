from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup 
import re 
import numpy as np
import pandas as pd

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Create function to predict sentiment in the String 
tokens = tokenizer.encode("This is a bad day. I hate this so much", return_tensors='pt')
result = model(tokens)
result.logits
print(int(torch.argmax(result.logits))+1)

r = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont') # URL of the website
soup = BeautifulSoup(r.text, 'html.parser') # Parse the HTML as a string
regex = re.compile('.*comment.*') # Regex to search for the class
results = soup.find_all('p', {'class':regex}) # Find all the tags with the class
reviews = [result.text for result in results] # Extract the text from the tags

df = pd.DataFrame(np.array(reviews), columns=['review']) # Create a dataframe with the reviews
df['review'].iloc[0] # Print the first review
def sentiment_score(review): 
    tokens = tokenizer.encode(review, return_tensors='pt') 
    result = model(tokens)
    return int(torch.argmax(result.logits))+1
print(sentiment_score(df['review'].iloc[1]))
df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
df['review'].iloc[3]