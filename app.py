from flask import Flask, render_template, jsonify, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('temp.html')

@app.route('/files')
def list_files():
    files = os.listdir('text')
    return jsonify(files)

@app.route('/text/<filename>')
def get_file_content(filename):
    try:
        with open(os.path.join('text', filename), 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return 'File not found', 404

if __name__ == '__main__':
    app.run(debug=True)
