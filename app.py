import os
import time
import torch
from flask import Flask, render_template, request
from transformers import RobertaForSequenceClassification, AutoTokenizer

# Enable Eager Execution
# tf.enable_eager_execution()
# tf.executing_eagerly()

# ------------------------------------- Load Sentiment Model -------------------------------------------
model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

# ------------------------------------- Build Web app ------------------------------------------
app = Flask(__name__)

def predict_sentiment(sentence):
    input_ids = torch.tensor([tokenizer.encode(sentence)])
    with torch.no_grad():
        output = model(input_ids)
    probabilities = torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0]
    labels = ['NEG', 'POS', 'NEU']
    sentiment = labels[probabilities.index(max(probabilities))]
    return sentiment, probabilities

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/sav')
def sav():
    return render_template('sav.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        sentence = request.form['inputSentence']
        start = time.time()

        # Predict sentiment using the function
        predicted_sentiment, probabilities = predict_sentiment(sentence)

        end = time.time()
        time2run = end - start

        # Pass the variables to the HTML template
        return render_template('result.html', predicted_sentiment=predicted_sentiment, sentence=sentence,
                               probabilities=probabilities, time2run=time2run)

    return "<h1>Please enter your paragraph in the text box!</h1>"

if __name__ == '__main__':
    app.run(debug=True)
