from flask import Flask, render_template, request
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
app = Flask(__name__)
def get_summary(context):
  LANGUAGE = "english"
  SENTENCES_COUNT = 2
  parser = PlaintextParser.from_string(context, Tokenizer(LANGUAGE))
  stemmer = Stemmer(LANGUAGE)
  summarizer = Summarizer(stemmer)
  summarizer.stop_words = get_stop_words(LANGUAGE)
  summary = str(summarizer(parser.document, SENTENCES_COUNT)[0:2])
  replaced = summary.replace('<Sentence:','')
  replaced1 = replaced.replace('>','')
  replaced2 = replaced1.replace('(','')
  replaced3 = replaced2.replace(')','')
  return replaced3

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        context = request.form.get("context")
        summary = get_summary(context)
        return render_template("index.html", context=context, summary=summary)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
