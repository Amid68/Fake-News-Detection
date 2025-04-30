#!/bin/bash

# Exit on any error
set -e

# Update package list and install dependencies
echo "Updating system and installing dependencies..."
apt-get update
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    make

# Install Docker
echo "Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
echo "Installing Docker Compose..."
DOCKER_COMPOSE_VERSION=v2.21.0
mkdir -p ~/.docker/cli-plugins/
curl -SL "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
ln -sf ~/.docker/cli-plugins/docker-compose /usr/local/bin/docker-compose

# Set up directories
echo "Setting up application directories..."
mkdir -p /opt/verifact/nginx/conf.d /opt/verifact/nginx/ssl /opt/verifact/logs /opt/verifact/backups

# Clone the repository (if URL provided)
if [ -n "$REPO_URL" ]; then
    echo "Cloning repository..."
    git clone "$REPO_URL" /opt/verifact
fi

# Set up environment variables
if [ ! -f /opt/verifact/.env ]; then
    echo "Creating .env file..."
    cp /opt/verifact/.env.docker /opt/verifact/.env
    # Generate a random secret key
    SECRET_KEY=$(openssl rand -hex 32)
    sed -i "s|SECRET_KEY=.*|SECRET_KEY=$SECRET_KEY|g" /opt/verifact/.env
fi

# Set up SSL certificates (if domain provided)
if [ -n "$DOMAIN" ]; then
    echo "Setting up SSL certificates for $DOMAIN..."
    # Install certbot
    apt-get install -y certbot

    # Get certificates
    certbot certonly --standalone -d "$DOMAIN" --non-interactive --agree-tos --email "$EMAIL" --no-eff-email

    # Copy certificates
    cp /etc/letsencrypt/live/"$DOMAIN"/fullchain.pem /opt/verifact/nginx/ssl/
    cp /etc/letsencrypt/live/"$DOMAIN"/privkey.pem /opt/verifact/nginx/ssl/

    # Update Nginx config
    sed -i "s|your-domain.com|$DOMAIN|g" /opt/verifact/nginx/conf.d/default.prod.conf
fi

# Set up cron job for SSL renewal
if [ -n "$DOMAIN" ]; then
    echo "Setting up SSL renewal cron job..."
    echo "0 3 * * * certbot renew --quiet && cp /etc/letsencrypt/live/$DOMAIN/* /opt/verifact/nginx/ssl/ && docker-compose -f /opt/verifact/docker-compose.yml -f /opt/verifact/docker-compose.prod.yml restart nginx" | crontab -
fi

# Start the application
echo "Starting application..."
cd /opt/verifact && docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

echo "Setup complete! VeriFact should now be running."
if [ -n "$DOMAIN" ]; then
    echo "Access your application at https://$DOMAIN"
else
    echo "Access your application at http://$(hostname -I | awk '{print $1}')"
fi