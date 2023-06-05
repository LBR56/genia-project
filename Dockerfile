FROM ubuntu:22.04

# https://opentutorials.org/module/2538/15818

RUN apt update -y && apt upgrade -y
RUN apt install git -y

WORKDIR /home/

RUN git clone https://github.com/LBR56/genia-project

WORKDIR /home/genia-project/
ENTRYPOINT ["/bin/sh", "-c", "/bin/bash"]
