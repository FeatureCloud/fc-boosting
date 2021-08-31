FROM python:3.8

RUN apt-get update
RUN apt-get install -y redis-server supervisor nginx

RUN pip3 install gunicorn

COPY requirements.txt /requirements.txt
RUN pip3 install -r ./requirements.txt

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY nginx/default /etc/nginx/sites-available/default
COPY docker-entrypoint.sh /entrypoint.sh

COPY . /app

EXPOSE 9000 9001

ENTRYPOINT ["sh", "/entrypoint.sh"]