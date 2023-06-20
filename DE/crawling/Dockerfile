FROM public.ecr.aws/lambda/python:3.9

COPY . .

ENV JAVA_HOME /usr/lib/jvm/java-1.7-openjdk/jre

RUN yum update -y
RUN yum install java-1.7.0-openjdk -y
RUN yum install -y gcc-c++
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/ssut/py-hanspell.git

CMD ["lambda_function.handler"]