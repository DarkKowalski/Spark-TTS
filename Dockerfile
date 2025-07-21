FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
WORKDIR /app

RUN conda create -n sparktts -y python=3.12
SHELL ["conda", "run", "-n", "sparktts", "/bin/bash", "-c"]
RUN echo "source activate sparktts" > ~/.bashrc

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
