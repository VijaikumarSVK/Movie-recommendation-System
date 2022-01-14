import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
df=pd.read_csv('name.csv')
df.shape
# create a list of important columns for the recommendation engine
columns=['Actors','Director','Genre','Title']
# check if there any missing values in the important columns
df[columns].isnull().sum()
# Create a function to combine the values of important values into a single string
def get_important_features(data):
  important_features=[]
  for i in range(0,data.shape[0]):
    important_features.append(data['Actors'][i]+''+data['Director'][i]+''+data['Genre'][i]+''+data['Title'][i])

  return important_features
df['important_features']=get_important_features(df)
df.head()
cm=CountVectorizer().fit_transform(df['important_features'])
# Get the cosine similarity matrix from the count matrix
cs=cosine_similarity(cm)
print(cs)
# Try the shape of the cosine similarity matrix[NXN]
cs.shape
# Get user input movie
title="Tusk"

# Find the movies id
movie_id=df[df.Title==title]['Movie_id'].values[0]
# Create a  list of enumerations for the similarity scores(Rank[0], Similarity Score[1])
scores=list(enumerate(cs[movie_id]))
# Sort the list 
# If we put X[0] then sorting will be based on rank rather than similarity score
sorted_scores=sorted(scores,key=lambda x: x[1],reverse=True)
sorted_scores=sorted_scores[1:]
# Print the recommendations in a sorted order
print(sorted_scores)
# print in right order
j=0
print("The 7 Most Recommended movies to",title," likes...")
for item in sorted_scores:
  movie_title=df[df.Movie_id==item[0]]['Title'].values[0]
  print(j+1,movie_title)
  j=j+1
  if j>6:
    break;
