FROM python:3.9.6-slim

RUN pip install -U update pip
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

# install wihtout create subenvironment
RUN pipenv install --system --deploy
COPY ["predict.py", "lin_reg.bin", "./"]

# listen for requests on the port
EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]