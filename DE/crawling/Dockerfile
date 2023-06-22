FROM public.ecr.aws/lambda/python:3.9

COPY . .

RUN yum update -y
RUN yum install -y tar gzip glibc wget gcc-c++ java-1.8.0-openjdk-devel python3 python3-devel python3-pip make diffutils curl git autoconf automake
RUN ln -s /sbin/ldconfig /bin/ldconfig

#RUN curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh > mecab.sh
# RUN pip install --upgrade --user setuptools
# RUN bash mecab.sh

WORKDIR /bin/

WORKDIR /tmp
RUN curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
RUN tar zxfv mecab-0.996-ko-0.9.2.tar.gz
WORKDIR /tmp/mecab-0.996-ko-0.9.2
RUN bash configure
RUN make
RUN make install

WORKDIR /tmp
RUN curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
RUN tar -zxvf mecab-ko-dic-2.1.1-20180720.tar.gz
WORKDIR /tmp/mecab-ko-dic-2.1.1-20180720
RUN bash autogen.sh
RUN bash configure
RUN bash -c 'echo "dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic" > /usr/local/etc/mecabrc'
RUN make install

WORKDIR /var/task/

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pip install git+https://github.com/ssut/py-hanspell.git

CMD ["lambda_function.handler"]