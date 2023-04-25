# Social Media Analysis Cheat Sheet
## Exp 1: Content Analysis
```python

from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd

df = pd.read_csv("./dataset")

stop_words = set(STOPWORDS)
def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in stop_words:
            result.append(str(token))
    return result

comments = df['comments'].apply(preprocess)
# OR
# newComments = []
# for i in (df.text):
#     newComments.append(preprocess(str(i)))
# df.comments=newComments
    
df.info()

# comments = df['comments'].apply(preprocess)
comments= df.text
dictionary = corpora.Dictionary(comments)
corpus = [dictionary.doc2bow(text) for text in comments]

num_topics = 10
passes = 10
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

for topic in lda_model.print_topics():
    print(topic)
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)
```

## Exp 2: Location Analysis
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px


counts = df["location"].value_counts()
[print(counts)]

plt.scatter(df.location.unique(),counts)
plt.bar(df.location.unique(),counts)
plt.pie(counts,labels=df.location.unique(),autopct='% 1.1f %%')
px.scatter(df, 'dislikes' , 'likes','location')
```

## Exp 3: Trend Analysis
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Read in the CSV file
df = pd.read_csv('trend_analysis.csv')
# Convert the 'date' column to a datetime data type
df['date'] = pd.to_datetime(df['date'])
print(df.head())

#You can do the following to split into day month and year:
df["day"] = df.date.dt.day()
df["month"] = df.date.dt.month()
df["year"] = df.date.dt.year()

# Set the 'date' column as the index of the DataFrame
df.set_index('date', inplace=True)
# Resample the data by day and count the number of entries in each day
# argument 'D' indicating that we want to resample by day.
daily_counts = df.resample('D').count()
print("Daily Counts: \n", daily_counts)
# Plot the daily counts over time
plt.plot(np.array(daily_counts.index), np.array(daily_counts['id']))
plt.xlabel('Day')
plt.ylabel('Number of Entries')
plt.title('Social Media Trends')
plt.show()


## Exp 4: Hashtag Analysis
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px


counts = df["hashtags"].value_counts()
[print(counts)]

plt.scatter(df.hashtags.unique(),counts)
plt.bar(df.hashtags.unique(),counts)
plt.pie(counts,labels=df.hashtags.unique(),autopct='% 1.1f %%')
```

## Exp5: Sentiment Analysis
```python
import pandas as pd
import numpy as np
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import nltk
nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

comments = df['comments']
comments = str(comments).encode('utf-8')

scoring = []
for com in df.comments:
    score = sia.polarity_scores(str(com))['compound']
    scoring.append(score)
df['score'] = scoring
# OR
# df['score'] = df['comments'].apply(lambda comments:sia.polarity_scores(str(comments))['compound'])
df['Sentiment'] = df['score'].apply(lambda s : 'Positive' if s > 0 else ('Neutral' if s == 0 else 'Negative'))
df.head()
df.Sentiment.value_counts()
positiveCount = df.Sentiment.value_counts()[0]
neutralCount = df.Sentiment.value_counts()[1]
negativeCount = df.Sentiment.value_counts()[2]
arr = [positiveCount,neutralCount,negativeCount]
labels = ["Positive", "Neutral", "Negative"]
plt.pie(arr, labels=labels, autopct='%1.1f%%')

pos_tweets = df[df["Sentiment"] == "Positive"]
neg_tweets = df[df["Sentiment"] == "Negative"]
sns.kdeplot(pos_tweets["score"], shade=True, label="Pos")
sns.kdeplot(neg_tweets["score"], shade=True, label="Neg")
plt.xlabel("Polarity Score")
plt.ylabel("Density")
plt.title("Sentiment Analysis of Tweets")
plt.legend()
plt.show()
```

## Exp 10: Community Detection using Girvan Newman
```python
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman

nodes = 25  
edges = 50  
random_seed= 20160
G = nx.gnm_random_graph(nodes,edges,seed=random_seed)

pos = nx.spring_layout(G, seed=random) 
nx.draw(G, pos=pos,with_labels = True)
plt.show()

comm = girvan_newman(G)
node_groups = []
for com in next(comm):
    node_groups.append(list(com))
print(node_groups)

color_map = []
for node in G:
    if node in node_groups[0]:
        color_map.append('blue')
    else: 
        color_map.append('green')  

nx.draw(G, node_color=color_map, with_labels=True)
plt.show()
```


## Exp 6: User Engagement Metrics
```
same as above ig
```

## Exp 7: EDA
```
Just make plots ggez
```

## Exp 8: Brand Analysis
```
Just analyze ggez - could be sentiment lol
```

## Exp 9: Analyze Competitor Activities
```
pata nai bhai pata nai - could also be sentiment lol
```
