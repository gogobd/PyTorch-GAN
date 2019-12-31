FROM nvidia/cuda:latest

# Install system dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
        wget \
        git \
    && apt-get clean

# Install python miniconda3 + requirements
ENV MINICONDA_HOME="/opt/miniconda"
ENV PATH="${MINICONDA_HOME}/bin:${PATH}"
RUN curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x Miniconda3-latest-Linux-x86_64.sh \
    && ./Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_HOME}" \
    && rm Miniconda3-latest-Linux-x86_64.sh

# COPY environment.yml environment.yml
# RUN conda env update -n=root --file=environment.yml
# RUN conda clean -y -i -l -p -t && \
#    rm environment.yml

# COPY requirements.txt requirements.txt
# RUN pip install requirements.txt

RUN conda install pytorch torchvision -c soumith && \
    conda install matplotlib && \
    conda install numpy && \
    conda install scipy && \
    conda install pillow && \
    conda install urllib3 && \
    conda install scikit-image && \
	conda install -c anaconda jupyter && \
	conda install -c conda-forge jupyterlab && \
	conda install -c anaconda nb_conda

RUN wget https://github.com/codercom/code-server/releases/download/1.408-vsc1.32.0/code-server1.408-vsc1.32.0-linux-x64.tar.gz && \
    tar -xzvf code-server1.408-vsc1.32.0-linux-x64.tar.gz && chmod +x code-server1.408-vsc1.32.0-linux-x64/code-server
 
# Copy
COPY . /PyTorch-GAN
WORKDIR /PyTorch-GAN

# Start container in notebook mode
CMD /code-server1.408-vsc1.32.0-linux-x64/code-server --allow-http --no-auth --data-dir / & \
    jupyter-lab --ip="*" --no-browser --allow-root
    