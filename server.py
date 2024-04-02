from flask import Flask, request
from datetime import datetime

app = Flask(__name__)

@app.route('/save_data', methods=['POST'])
def save_data():
    # Get the data from the request
    data = request.get_data()

    # Convert bytes to string
    data_str = data.decode('utf-8')
    
    print(data_str)

    # Get current time and format it as a string
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save the data to a file with current time in its name
    with open(f'Data_{current_time}.txt', 'w') as f:
        f.write(data_str)

    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)