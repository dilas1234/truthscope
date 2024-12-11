# routes.py (Flask Routes)

from flask import Flask, request, jsonify
from models.text_verification import verify_text
from models.image_verification import verify_image
from models.source_credibility import score_credibility

app = Flask(__name__)

@app.route('/verify_text', methods=['POST'])
def verify_text_route():
    text = request.json.get('text')
    result = verify_text(text)
    return jsonify({'result': result})

@app.route('/verify_image', methods=['POST'])
def verify_image_route():
    image_path = request.json.get('image_path')
    result = verify_image(image_path)
    return jsonify({'result': result})

@app.route('/score_credibility', methods=['POST'])
def score_credibility_route():
    source_data = request.json
    score = score_credibility(source_data)
    return jsonify({'credibility_score': score})

if __name__ == '__main__':
    app.run(debug=True)
