FROM python:3
ADD data/ /data/
ADD modules/*.py /modules/
ADD mnist.py /
RUN pip install requests
CMD [ "python3", "-u", "./mnist.py" ]
