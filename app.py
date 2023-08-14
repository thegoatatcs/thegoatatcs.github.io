from flask import Flask, render_template, request
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)

def load_data():
    final_data = pd.read_csv('final.csv')
    final_data = final_data.loc[:, ~final_data.columns.str.contains('^Unnamed')]
    return final_data

def train_model():
    final_data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(final_data['essay_text'], final_data['label'], test_size=0.15)
    
    pipe = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('model', LogisticRegression())
    ])

    model = pipe.fit(X_train, y_train)
    return model

def predict(model, input_text):
    prediction = model.predict([input_text])
    prob = model.predict_proba([input_text])
    return prediction[0], prob[0]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        if input_text:
            model = train_model()
            prediction, prob = predict(model, input_text)
            return render_template("index2.html", input_text=input_text, prediction=prediction, prob=prob)
        else:
            error_message = "Please enter some text."
            return render_template("index2.html", error_message=error_message)
    return render_template("index2.html")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
