from flask import Flask, request, jsonify, render_template
import numpy as np
from doc_similarity_detector import document_path_similarity

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compare', methods = ['POST'])
def review():
    text1 = request.form['text1']
    text2 = request.form['text2']
    
    similarity_score = document_path_similarity(text1, text2)
    app.logger.info('similarity_score %s' , similarity_score)    
   
    if np.isnan(similarity_score):
        similarity_score = 0
    
    return render_template('result.html', text1=text1, text2=text2, similarity_score=similarity_score)


if __name__ == "__main__":
    app.run(debug=True)