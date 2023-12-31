FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN apt update \
    && apt install -q -y --no-install-recommends --no-install-suggests gcc nginx certbot python3-pip python3-venv python3-certbot-nginx \
    && python3 -m venv venv


WORKDIR /tmp

COPY ./requirements.txt requirements.txt
RUN pip install tensorflow \
    && pip install gunicorn \
    && pip --default-timeout=3600 install -r requirements.txt


WORKDIR /var/www
COPY . .

RUN apt autoremove \
    && rm -rf /tmp/*

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "skripsi.wsgi:application"]