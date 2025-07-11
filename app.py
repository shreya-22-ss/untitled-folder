from flask import Flask, render_template, request, jsonify
from datetime import datetime
import subprocess
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/market')
def market():
     return render_template('market.html')

@app.route('/farmer')
def farmer():
     return render_template('farmer.html')

@app.route('/customer')
def customer():
     return render_template('customer.html')


@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    print("Received request for recommendation")
    data = request.json
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    district = data.get('district')

    if not (latitude and longitude and district):
        return jsonify({'error': 'Missing latitude, longitude, or district'}), 400

    current_date = datetime.now().strftime('%Y-%m-%d')

    try:
        result = subprocess.check_output([
            sys.executable, 'market_integrated.py',
            str(latitude), str(longitude), str(district), str(current_date)
        ], universal_newlines=True)
        print("Result from market_integrated.py:")
        print(result)
    except Exception as e:
        print("Error running market_integrated.py:", e)
        return jsonify({'error': str(e)}), 500

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)