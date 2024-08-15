# Setup base image.
FROM muazhari/cuda-torch-tensorflow:latest

# Setup workdir.
ENV WORKDIR=/workdir

WORKDIR $WORKDIR

# SSH server setup.
RUN apt update -y

RUN yes | DEBIAN_FRONTEND=noninteractive apt install \
    openssh-server \
    git

RUN mkdir /var/run/sshd

RUN echo 'root:root' | chpasswd

RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Setup dependencies.
COPY requirements.txt .

RUN pip3 install -r requirements.txt --use-feature=fast-deps

RUN pip3 install ray[default,tune] --use-feature=fast-deps
