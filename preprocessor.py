import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
def preprocess(resumeText):
    # Convert to lowercase
    resumeText = resumeText.lower()

    # Remove non-alphabetic characters and keep only letters and whitespaces
    resumeText = re.sub(r'[^a-zA-Z\s]', '', resumeText)

    # Remove URLs
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)

    # Remove 'RT' and 'cc'
    resumeText = re.sub(r'RT|cc', ' ', resumeText)

    # Remove punctuations
    resumeText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', resumeText)

    # Remove non-ASCII characters
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)

    # Remove extra whitespaces
    resumeText = re.sub(r'\s+', ' ', resumeText)

    # Tokenize the text
    words = word_tokenize(resumeText)

    # Remove stop words and lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Join the words back into a clean resume text
    cleaned_resume = ' '.join(words)

    return cleaned_resume
