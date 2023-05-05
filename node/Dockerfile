FROM python:3.7

RUN python -m pip install flask flask-cors gunicorn numpy tensorflow pillow

WORKDIR /app

ADD myapp.py myapp.py
ADD image.py image.py

EXPOSE 8000

CMD [ "gunicorn", "--bind", "0.0.0.0:8000", "myapp:app" ]