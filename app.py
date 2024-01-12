from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K

app = Flask(__name__)

# Load Siamese LSTM model
model_path = "Model\weights.11-0.189.h5"  # Replace with the actual path to your model file

def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

malstm = load_model(model_path, custom_objects={'exponent_neg_manhattan_distance': exponent_neg_manhattan_distance})
maxlen = 60  # 
threshold = 0.69

def preprocess_text(text):
    tokenizer = Tokenizer(num_words=5000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

outtolabel = {0 : "No, The submitted sentences are not paraphrase", 
        1 : "Yes, The submitted sentences are paraphrase"}

def predict(encod1, encod2):
    encod1 = preprocess_text([encod1])
    encod2 = preprocess_text([encod2])
    prediction = malstm.predict([encod1, encod2])
    print("encod1:", encod1)
    print("encod2:", encod2)
    print("prediction:", prediction)
    return outtolabel[int(prediction[0][0] > threshold)]  # Adjust the threshold as needed

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

# @app.route("/about")
# def about_page():
#     return render_template("about.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        sentence1 = request.form['sentence_1']
        sentence2 = request.form['sentence_2']
        label = predict(sentence1, sentence2)

    return render_template("index.html", prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
