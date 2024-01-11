from flask import Flask, render_template,request
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm



app = Flask(__name__)
emodel = SentenceTransformer('paraphrase-MiniLM-L6-v2')
threshold = 0.69
outtolabel = {0 : "No, The submitted sentences are not paraphrase", 
        1 : "Yes, The submitted sentences are paraphrase"}

def predict(encod1 , encod2):
    encod1 = emodel.encode(encod1)
    encod2 = emodel.encode(encod2)
    cos_sim = np.dot(encod1,encod2)/(norm(encod2)*norm(encod1))
    if (cos_sim>threshold):
        return 1
    else:
        return 0

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        sentence1 = request.form['sentence_1']
        sentence2 = request.form['sentence_2']
        label = predict(sentence1,sentence2)
        

    return render_template("index.html", prediction = outtolabel[label])


if __name__ =='__main__':
    app.run(debug = True)



