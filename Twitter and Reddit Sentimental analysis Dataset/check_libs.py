import sys
try:
    import pandas
    print("pandas: OK")
except ImportError:
    print("pandas: MISSING")

try:
    import nltk
    print("nltk: OK")
except ImportError:
    print("nltk: MISSING")

try:
    import textblob
    print("textblob: OK")
except ImportError:
    print("textblob: MISSING")

try:
    import wordcloud
    print("wordcloud: OK")
except ImportError:
    print("wordcloud: MISSING")

try:
    import tensorflow
    print("tensorflow: OK")
except ImportError:
    print("tensorflow: MISSING")
