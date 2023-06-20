FROM public.ecr.aws/lambda/python:3.9

COPY . .

ENV JAVA_HOME /usr/lib/jvm/java-1.7-openjdk/jre

RUN apt-get update && apt-get install -y g++ default-jdk
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/ssut/py-hanspell.git

CMD ["lambda_function.handler"]