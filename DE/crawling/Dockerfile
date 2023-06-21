FROM public.ecr.aws/lambda/python:3.9

COPY . .

RUN yum update -y
RUN yum install -y gcc-c++ java-1.8.0-openjdk-devel python3 python3-devel python3-pip make diffutils
RUN python3 -m pip install --upgrade pip

RUN pip install -r requirements.txt
RUN yum install -y curl git
RUN curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh
RUN pip install git+https://github.com/ssut/py-hanspell.git

CMD ["lambda_function.handler"]