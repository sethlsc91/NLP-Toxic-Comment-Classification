## DABPT01 - General Assembly Singapore - Capstone Project - Classification of Toxic Social Media Comments
## CONTENT DISCLAIMER
Some sections of the following content, while constructed for academic purposes, might be considered offensive, controversial, or triggering to some viewers - if you are uncomfortable with viewing such content, please do turn away.
#### Themes include, but are not limited to:
- Violence
- Discrimination
- Sexually suggestive / explicit content
- Strong language

## Introduction & Project Overview
This repository is home to my capstone project for **General Assembly Singapore: Data Analytics Bootcamp**, attempting classification of toxic social media comments by way of Natural Language Processing (NLP) modeling techniques (ie. problematic VS non-problematic).

The rise of online platforms has significantly transformed the way individuals interact, share ideas, and express opinions. While this digital evolution has offered numerous benefits, it has also given rise to a slew of harmful behaviours, including the spread of toxic and offensive comments. The anonymity provided by the internet often emboldens users to post inflammatory, abusive, or otherwise inappropriate content without fear of direct consequences. This growing prevalence of online toxicity poses serious dangers, ranging from psychological distress among victims to broader social polarization and real-world repercussions.

In light of these challenges, this project explores the application of NLP modeling techniques to identify and manage toxic comments on social media. By leveraging machine learning models and linguistic analysis, the goal is to understand the characteristics of problematic online language and investigate how it can be effectively detected and mitigated. The project aims not only to highlight the severity of unregulated toxic discourse, but also to contribute toward developing automated solutions that can support safer, more respectful digital communities.

Potential applications of this project span a wide range of digital platforms where user-generated content is prevalent:

- Online message boards
- Social media comment sections
- Chat platforms
- Real-time chatbot interactions

By integrating these tools into content moderation workflows, platforms can proactively identify and flag harmful language before it escalates. Additionally, such models can support content filtering in educational forums, gaming communities, and public review sites, ensuring that discussions remain civil and inclusive.

## Dataset
For the purpose of this project, the dataset used (train.csv) contains **159,571** comments, and can be obtained from [Kaggle](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge/data).

A quick glance at some of the comments present in the data suggests that they were scraped from Wikipedia, a free online encyclopedia written and maintained by anyone and everyone in the world.

The data dictionary is as follows - this will provide some context to how the data is being analysed and interpreted:

| Column Header | Data Type | Description |
| --- | --- | --- |
| id | _obj_ | Unique identifier for online comment |
| comment_text | _obj_ | Online comment in full |
| toxic | _int_ | Binary toxicity flag if comment deemed toxic ([1]: Yes, [0]: No) |
| severe_toxic | _int_ | Binary toxicity flag if comment deemed very toxic ([1]: Yes, [0]: No) |
| obscene | _int_ | Binary toxicity flag if comment deemed obscene ([1]: Yes, [0]: No) |
| threat | _int_ | Binary toxicity flag if comment deemed threatening ([1]: Yes, [0]: No) |
| insult | _int_ | Binary toxicity flag if comment deemed insulting ([1]: Yes, [0]: No) |
| identity_hate | _int_ | Binary toxicity flag if comment deemed slanderous to one's identity ([1]: Yes, [0]: No) |

## Summary of Exploratory Data Analysis (EDA)
Various pre-processing measures were first applied to the raw comment strings found in the dataset to ensure fair analysis of all eligible words during modelling:

- Lowercasing
- Punctuation removal
- Number removal
- Stop word removal
- Whitespace normalisation
- Tokenisation
- Lemmatisation

Among all **159,571** comments analysed, **~10%** of them (16,225) had at least one (1) of the aforementioned toxicity flags attributed to them. Also, **~94%** of these problematic comments (15,294) had the _'toxic'_ flag marked against them, suggesting that usage of this flag is a baseline for a problematic comment to be deemed so.

![Image](https://github.com/user-attachments/assets/22d1bebe-604f-48cc-8a44-49a42992e701)

![Image](https://github.com/user-attachments/assets/9f5da9c5-b734-41a5-bc8b-c28bb98e4da1)

![Image](https://github.com/user-attachments/assets/a45cc5fa-ae7f-46d8-9184-f58c4fbff0e0)

![Image](https://github.com/user-attachments/assets/96427cd9-8431-471b-b796-5bb91670ece4)

Separate word clouds generated for four (4) toxicity flags (_'toxic'_, _'severe_toxic'_, _'obscene'_, _'insult'_) showcase the most common words found in each of those flags - the bigger the size of the word in the cloud, the higher the frequency. It is observed that certain high-frequency words (eg. _'fuck'_, _'shit'_, _'suck'_) appear across all the word clouds.

![Image](https://github.com/user-attachments/assets/a045f62a-7768-4e72-83eb-de9bac808092)

Venn diagrams plotted between _'toxic'_ and the other five (5) possible toxicity flags also suggest a high-degree of overlap, particularly for comments marked with two (2) or more toxicity flags (where one (1) of them is _'toxic'_). In the context of this project, this observation indicates a diminishing value of multi-labeling for the following reasons:

- The project target is to apply swift corrective action against lingering problematic comments in the public sphere by detecting offensive words in a given comment string
- Since ~94% of problematic comments are marked _'toxic'_, it appears that there is little value to add more than 1 toxicity flag to any given comment string since the resulting corrective action is the same for any comment string with 1 or more toxicity flags (ie. removal from public sphere)

**With this in mind, a new simplified flag - _'flagged'_ - was created, and the project focus is to discern, broadly, problematic comments from the non-problematic ones:**

| Column Header | Data Type | Description |
| --- | --- | --- |
| flagged | _int_ | Binary flag if comment deemed problematic ([1]: Yes, [0]: No) |

## Vectorisers / Models of Choice

Two (2) vectorisers and two (2) models were used for this project, totalling to four (4) possible vectoriser-model combinations:

| Vectoriser | Model | Shorthand Reference |
| --- | --- | --- |
| Count | Logistic Regression | cvec_logr |
| Count | Multinomial NB | cvec_nb |
| TF-IDF | Logistic Regression | tvec_logr |
| TF-IDF | Multinomial NB | tvec_nb |

By applying an (80:20) train-test split on the dataset and putting it through all four (4) vectoriser-model combinations, some key metric results are as follows:

| Metric | cvec_logr | cvec_nb | tvec_logr | tvec_nb |
| --- | --- | --- | --- | --- |
| Recall | 68.04% | 74.58% | 62.10% | 16.46% |
| Precision | 85.09% | 73.07% | 92.01% | 99.26% |
| F1-Score | 75.62% | 73.81% | 74.15% | 28.23% |

The above statistical metrics are explained in the context of this project below:
- **Recall score** measures the proportion of known problematic comments being incorrectly flagged as innocent - the higher the percentage, the better the vectoriser-model is at flagging out problematic comments
- **Precision score** measures the proportion of innocent comments being incorrect flagged as problematic - the lower the percentage, the more likely the vectoriser-model is at flagging out innocent comments as problematic
- **F1-Score** represents the harmonic mean between Recall and Precision scores, and ensures that one does not place too much emphasis on either Recall or Precision scores when identifying the most suitable vectoriser-model

Given that the aim of this project is to identify, effectively, problematic comments and remove them from public online platforms (where applicable) as soon as possible, one can infer that the **Count Vectoriser - Multinomial NB combination (cvec_nb)** performed best at predicting problematic comments due to it having achieved the highest Recall score and a competitive F1-Score metric.

## Analysis Limitations
![Image](https://github.com/user-attachments/assets/50c30bcb-ce8a-48da-b7dc-0d5ce3ca11c5)
- It was observed that some of the most frequent words found amongst _'flagged'_ comments are seemingly innocent words such as '_wikipedia'_, _'as'_ and _'people'_, possibly due to the usage of a default stop word library during the analysis (**nltk.corpus.stopwords** for Python)
- Analysis of each eligible word / token did not factor in the context in which the word was used in the comment string (eg. the word _'cut'_ can be used as an action word for either progressive work or self-harm)
- Uni-gram analysis was employed during this project, where each word / token was analysed independently and word order in a comment string was not considered
- There were instances of delibrate misspelling in comment strings to circumvent existing online safety restrictions - after having applied the pre-processing measures to streamline all words / tokens for analysis, the result of these words might be unrecoginisable, and treated as a completely new word during modeling

## Future Work

- Further customisation / increase of the stop word list for future analyses can be looked into, so that analysis of offensive words could be narrowed down further
- The models used in the present analysis were the most fundamental and commonly-used (Logistic Regression / Multinomial NB) - given the multi-label nature of this dataset, more sophisticated models or techniquees could be employed to analyse words / tokens in accordance to the specified toxicity flag in more detail
- Employ higher _n_-gram analysis (ie. bi-gram, tri-gram) to incorporate word order / context into modeling, so that resulting predictions could be more accurate 
