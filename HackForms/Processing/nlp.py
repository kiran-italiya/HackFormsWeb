from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import regex as re

# nltk.download('vader_lexicon')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
tricky_sentences = [
"Most automated sentiment analysis tools are shit.",
"VADER sentiment analysis is the shit.",
"Sentiment analysis has never been good.",
"Sentiment analysis with VADER has never been this good.",
"Warren Beatty has never been so entertaining.",
"I won't say that the movie is astounding and I wouldn't claim that \
the movie is too banal either.",
"I like to hate Michael Bay films, but I couldn't fault this one",
"It's one thing to watch an Uwe Boll film, but another thing entirely \
to pay for it",
"The movie was too good",
"This movie was actually neither that funny, nor super witty.",
"This movie doesn't care about cleverness, wit or any other kind of \
intelligent humor.",
"Those who find ugly meanings in beautiful things are corrupt without \
being charming.",
"There are slow and repetitive parts, BUT it has just enough spice to \
keep it interesting.",
"The script is not fantastic, but the acting is decent and the cinematography \
is EXCELLENT!",
"Roger Dodger is one of the most compelling variations on this theme.",
"Roger Dodger is one of the least compelling variations on this theme.",
"Roger Dodger is at least compelling as a variation on the theme.",
"they fall in love with the product",
"but then it breaks",
"usually around the time the 90 day warranty expires",
"the twin towers collapsed today",
"However, Mr. Carter solemnly argues, his client carried out the kidnapping \
under orders and in the ''least offensive way possible.''"
 ]
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
stop_words.update(['mrs.','ms.','mr.','``','.', ',', '"',"''", "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
for sentence in tricky_sentences:
    print(sentence+'\n')
    sentence = re.sub('[^a-zA-Z]', ' ', sentence) # punctuations removal
    # text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text) # tags removal
    sentence = re.sub("(\\d|\\W)+", " ", sentence) #removal of special chars and digits

    tokens = word_tokenize(sentence.lower())
    tokens_stop_words_removed = [i for i in tokens if i not in stop_words]
    print(tokens_stop_words_removed)
    print('Pos \n',nltk.pos_tag(tokens_stop_words_removed))
    ss = sid.polarity_scores(sentence)
    # print(ss)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print('\n')


