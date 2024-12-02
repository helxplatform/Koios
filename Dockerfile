######################################################
#
# A container for the core semantic-search capability.
#
######################################################
FROM python:3.12.1-slim-bullseye


# Install required packages
RUN apk update && \
   apk add g++ gcc \ 
# upgrade openssl \

# RUN apk add openssl=3.1.4-r5


RUN pip install --upgrade pip
# Create a non-root user.
ENV USER koios
ENV HOME /home/$USER
ENV UID 1000

RUN adduser --disabled-login --home $HOME  --uid $UID $USER

USER $USER
WORKDIR $HOME

ENV PATH=$HOME/.local/bin:$PATH

# Copy over the source code
RUN mkdir koios
COPY --chown=$USER . koios/
WORKDIR $HOME/koios
ENV PYTHONPATH=$HOME/koios/src
RUN pip install -r requirements.txt
ENTRYPOINT python src/server.py

