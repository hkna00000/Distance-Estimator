from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app) 
FASTAPI_URL = "http://127.0.0.1:8000/process"  # URL of the FastAPI service

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        print("Received data:", data)  # Log the received data
        
        # Forward to FastAPI
        response = requests.post(FASTAPI_URL, json=data)
        print("FastAPI response status:", response.status_code)  # Log FastAPI response
        print("FastAPI response body:", response.text)  # Log response text
        
        if response.status_code == 200:
            result = response.json().get("result", "No result")
            return jsonify({"result": result})
        else:
            return jsonify({"error": "Error from FastAPI service"}), 500
    except Exception as e:
        print("Error occurred:", str(e))  # Log the error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

