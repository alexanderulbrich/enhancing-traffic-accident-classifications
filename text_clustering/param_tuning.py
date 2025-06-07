from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
import pandas as pd
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from itertools import product
from sklearn.metrics import pairwise_distances
import json
from tqdm import tqdm

# Load embeddings and texts
embeddings = pd.read_csv("data/encoded_unfalltext_first50000.csv")
embeddings = embeddings.to_numpy()
text = pd.read_csv("data/D_Unfalltext_image_50000_subsample.csv")["Text"]
text = text.to_numpy()

# download stopwords and extract German words
nltk.download("stopwords")
stop_words = stopwords.words("german")

# Those are variables which stay as they are
embedding_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
representation_model = MaximalMarginalRelevance()

# Define Function which takes the hyperparams as inputs and returns metrics (coherence, diversity, number of topics and number of outliers)
def calculate_metrics(n_neighbors, n_components, min_dist, min_cluster_size, text):
  # define umap model based on hyperparams
  umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric='manhattan', low_memory=False, force_approximation_algorithm=True, random_state=42)
  # define hdbscan model based on hyperparams
  hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='manhattan', cluster_selection_method='eom', prediction_data=True)
  # define vectorizer model, this is always the same and we exclude german stopwords for the representations
  vectorizer_model = CountVectorizer(stop_words=stop_words)
  # specicy and fit topic model
  topic_model = BERTopic(
     embedding_model=embedding_model,
     representation_model=representation_model,
     umap_model=umap_model,
     hdbscan_model=hdbscan_model,
     nr_topics="auto",
     ctfidf_model=ctfidf_model,
     vectorizer_model=vectorizer_model)
  topic_model.fit_transform(text, embeddings)
  # apply preprocess function (removes \n and \t and only keeps alpha numerical characters)
  cleaned_docs = topic_model._preprocess_text(text)
  # take vectorizer model from above (includes stopwords)
  vectorizer = topic_model.vectorizer_model
  # analyzer is from sklearn and is responsible for tokenizing and preprocessing text according to our vectorizer above
  analyzer = vectorizer.build_analyzer()
  # apply to each document in cleaned_docs
  tokens = [analyzer(doc) for doc in cleaned_docs]
  # create a gensim dictionary from tokenized documents (map each unique token to a unique int ID)
  dictionary = corpora.Dictionary(tokens)
  # now create bag-of-words representation using dictionary above
  corpus = [dictionary.doc2bow(token) for token in tokens]
  # get topics from out topic model
  topics = topic_model.get_topics()
  # count topics but we don't consider outliers a topic
  num_topics = len(set(topics)) - 1 
  # sometimes, when we only have 1 topic and outliers this can be 1 or even 0. So we do a case distinction here
  if num_topics <= 1:
    # extract topic representations
    topic_words = [[word for word, _ in topic_model.get_topic(topic)] for topic in topics.keys()]
    # remove empty strings when mapped to dictionary
    topic_words_clean = [[word for word in topic if word != '' and word in dictionary.token2id] for topic in topic_words]
    topic_words_clean = [topic for topic in topic_words_clean if len(topic) > 0]
    # specify coherence model, pass arguments: extracted toppics, tokens, corpus is the BoW representation, dictionary maps IDs to words and coherence=c_v specifies coherence measure (document co-coherence)
    coherence_model = CoherenceModel(topics=topic_words_clean, texts=tokens, corpus=corpus, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
  else:
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] for topic in range(len(set(topics))-1) if topic != -1]
    topic_words_clean = [[word for word in topic if word != '' and word in dictionary.token2id] for topic in topic_words]
    topic_words_clean = [topic for topic in topic_words_clean if len(topic) > 0]
    coherence_model = CoherenceModel(topics=topic_words_clean, texts=tokens, corpus=corpus, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
  # ccreate a set to contain all topic representations and map to iterating number
  all_words = set(word for topic in topic_words for word in topic)
  word_to_index = {word: i for i, word in enumerate(all_words)}
  binary_matrix = np.zeros((len(topic_words), len(all_words)), dtype=int)
  # add 1 to 0 matrix to calculate jaccard distance
  for i, topic in enumerate(topic_words):
     for word in topic:
        if word in word_to_index:
           binary_matrix[i, word_to_index[word]] = 1

  topic_diversity = pairwise_distances(binary_matrix, metric='jaccard')

  num_topics = len(set(topics)) - (1 if -1 in topics else 0)
  num_outliers = topic_model.get_topic_freq(-1)

  n = len(topic_words)

  diversity_score = 0

  if n > 1:
    if np.isnan(topic_diversity).any() or np.isinf(topic_diversity).any():
       raise ValueError("topic_diversity contains invalid values.")
    diversity_score = np.sum(np.triu(topic_diversity, k=1)) / (n * (n-1) / 2)

  return [f"Coherence: {coherence_score}", f"Diversity: {diversity_score}", f"N of topics: {num_topics}", f"N of outliers: {num_outliers}",]

param_grid = {
   "n_neighbors": [10],
   "n_components": [5],
   "min_dist": [0.0],
   "min_cluster_size": [300]
}

param_combinations = list(product(
   param_grid["n_neighbors"],
   param_grid["n_components"],
   param_grid["min_dist"],
   param_grid["min_cluster_size"]
))

results = []

for params in tqdm(param_combinations, desc="Trying different parameters"):
    n_neighbors, n_components, min_dist, min_cluster_size = params
    try:
       coherence = calculate_metrics(
          n_neighbors=n_neighbors,
          n_components=n_components,
          min_dist=min_dist,
          min_cluster_size=min_cluster_size,
          text=text
          )
       
       results.append({
          "n_neighbors": n_neighbors,
          "n_components": n_components,
          "min_dist": min_dist,
          "min_cluster_size": min_cluster_size,
          "coherence": coherence[0].split(": ")[1],
          "diversity": coherence[1].split(": ")[1],
          "n_topics": coherence[2].split(": ")[1],
          "n_outliers": coherence[3].split(": ")[1]
          })
       
    except Exception as e:
       print(f"Error with params {params}: {e}")
       results.append({
          "n_neighbors": n_neighbors,
          "n_components": n_components,
          "min_dist": min_dist,
          "min_cluster_size": min_cluster_size,
          "error": str(e)
          })

output_file_json = "data/param_results.json"
with open(output_file_json, "w") as json_file:
   json.dump(results, json_file, indent=4)









