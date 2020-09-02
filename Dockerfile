FROM python:3
ADD data/ /data/
ADD modules/*.py /modules/
ADD results/ /results/
ADD mnist.py /
RUN pip3 install argparse matplotlib numpy torch torchvision
CMD [ "python3", "-u", "./mnist.py" ]
