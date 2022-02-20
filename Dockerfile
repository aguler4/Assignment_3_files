FROM python:3.8

COPY . ./

RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader all

CMD ["Assignment_3_Adarsh_Guler.py"]

ENTRYPOINT ["python"]

