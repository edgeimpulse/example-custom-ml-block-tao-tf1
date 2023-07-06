FROM nvcr.io/ea-tlt/tao_ea/tao-toolkit:5.0.0-ea-tf1.15.5

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY . ./

ENTRYPOINT ["/bin/bash", "/app/run.sh"]
