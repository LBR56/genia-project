from m_config import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY
from m_config import AWS_S3_BUCKET_NAME, AWS_S3_BUCKET_REGION
from flask import Flask
import boto3
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
s3 = boto3.client(
    's3',
    region_name=AWS_S3_BUCKET_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

@app.route('/')
def index():
    # S3 버킷에서 HTML 파일 다운로드
    bucket_name = AWS_S3_BUCKET_NAME
    file_name = 'index.html'
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    html_content = response['Body'].read().decode('utf-8')

    # 로그 출력
    app.logger.info('HTML 파일을 성공적으로 다운로드하였습니다.')

    return html_content

if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 5000)
    # print(index())
