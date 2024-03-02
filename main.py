import cv2
from flask import Flask, jsonify, request, render_template
import base64

import cv2 as cv  
import numpy as np  

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect_forgery', methods=['POST'])
def detect_forgery():
    image1 = request.files['image1']
    image2 = request.files['image2']

    img1_bytes = image1.read()
    img2_bytes = image2.read()

    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

    img1 = cv2.resize(img1, (300, 200))
    img2 = cv2.resize(img2, (300, 200))

   
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Encode image to Base64 format
    _, img_data = cv2.imencode('.jpg', img3)
    img_str = base64.b64encode(img_data).decode('utf-8')

    # Display the matched image

 
    # Calculate average distance of matches
    distances = [match.distance for match in matches]
    avg_distance = sum(distances) / len(distances)

    if avg_distance == 0:
        result = "Result - The test image is Not forged"
    else:
        result = "Result - The test image is Forged"

    return render_template('result.html', result=result, img_data=img_str)

if __name__ == '__main__':
    app.run(debug=True)
