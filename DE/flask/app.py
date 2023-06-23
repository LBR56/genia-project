from flask import Flask, render_template
import boto3

app = Flask(__name__)

s3 = boto3.resource('s3')
bucket_name = 'youtube-s3-bucket'  # S3 버킷 이름

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    s3.Bucket(bucket_name).upload_file('index.html', 'index.html')
    return 'File uploaded successfully!'

if __name__ == '__main__':
    app.run(debug=True)
