# Setup base image.
FROM muazhari/cuda-torch-tensorflow:latest

# Setup workdir.
ENV WORKDIR=/workdir
WORKDIR $WORKDIR

# Install apt dependencies.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-fast update -y \
    && \
    yes | apt-fast install -y \
    openssh-server \
    git

# Setup sshd.
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Copy the requirements.txt file to the working directory.
COPY ./requirements.txt ./requirements.txt

# Install python dependencies from requirements.txt.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    ray[default,tune] \
    -r requirements.txt \
    --break-system-packages \
    --no-build-isolation
