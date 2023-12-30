# django-skripsi
Django project untuk skripsi

# setup VPS
## update dan upgrade
`sudo apt update
sudo apt upgrade`

## Install package
`sudo apt install nginx python3-venv certbot python3-certbot-nginx`

## clone project
`cd /var/www/
sudo git clone https://github.com/sebastianbk98/django-skripsi.git
sudo chmod -R a+rwx /var/www/django-skripsi
cd django-skripsi`

## Python Virtual Environment
`python3 -m venv venv
source venv/bin/activate`

## Install Python Package
`pip install gunicorn
pip --default-timeout=3600 install -r requirements.txt`

## Update Django Settings
Change SMTP

## Run Gunicorn
`screen -S gunicorn_session`
`gunicorn --timeout 3600 --worker 3 --bind 0.0.0.0:8000 skripsi.wsgi:application`
`screen -r gunicorn_session`

## create SSL Cert
`sudo certbot --nginx -d analisissentimen-digitalkorlantas.my.id -d www.analisissentimen-digitalkorlantas.my.id`

## setup Nginx
`sudo nano /etc/nginx/sites-available/django-skripsi`
```
server {
    listen 80;
    server_name analisissentimen-digitalkorlantas.my.id, , www.analisissentimen-digitalkorlantas.my.id;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name analisissentimen-digitalkorlantas.my.id, www.analisissentimen-digitalkorlantas.my.id;
    client_max_body_size 50M;
    
    ssl_certificate /etc/letsencrypt/live/analisissentimen-digitalkorlantas.my.id/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/analisissentimen-digitalkorlantas.my.id/privkey.pem;

    # Other SSL configurations (e.g., SSL protocols, ciphers, etc.) can be added here
    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers 'EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH';
    
    location /static/ {
        alias /var/www/django-skripsi/staticfiles/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;  # Assuming Gunicorn is running on this port
        proxy_connect_timeout 3600s; # Adjust the connection timeout (default: 60s)
        proxy_send_timeout 3600s; # Adjust the send timeout
        proxy_read_timeout 3600s; # Adjust the read timeout
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Optional: Additional headers for WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```
`sudo ln -s /etc/nginx/sites-available/django-skripsi /etc/nginx/sites-enabled/`

## Test Nginx and restart
`sudo nginx -t`
`sudo systemctl restart nginx`

## Setup firewall
`sudo ufw enable`
`sudo ufw allow 'Nginx Full'`
`sudo ufw status`
