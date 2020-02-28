from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import re

# nltk.download('vader_lexicon')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
def do_nlp(sentence):
    print(sentence + '\n')
    sid = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(['mrs.', 'ms.', 'mr.', '``', '.', ',', '"', "''", "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    return ss['compound']