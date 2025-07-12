#!/bin/bash

# Oracle Cloud Compute Instance Deployment Script
# Replace the variables below with your actual values

# Configuration - UPDATE THESE VALUES
ORACLE_IP="150.136.76.172"
ORACLE_USER="ubuntu"
PROJECT_NAME="crop-recommendation-app"
LOCAL_PROJECT_PATH="/Users/shreya/Documents/untitled folder"

echo "🚀 Starting Oracle Cloud deployment..."

# Check if SSH key is available
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "❌ SSH key not found. Please ensure you have SSH access to your Oracle instance."
    echo "You can generate an SSH key with: ssh-keygen -t rsa -b 4096"
    exit 1
fi

# Test SSH connection
echo "🔍 Testing SSH connection to Oracle Cloud instance..."
ssh -o ConnectTimeout=10 -o BatchMode=yes $ORACLE_USER@$ORACLE_IP exit 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Cannot connect to Oracle Cloud instance. Please check:"
    echo "   - Your instance IP address: $ORACLE_IP"
    echo "   - SSH key is added to your instance"
    echo "   - Security list allows SSH (port 22)"
    exit 1
fi

echo "✅ SSH connection successful!"

# Create deployment package
echo "📦 Creating deployment package..."
cd "$LOCAL_PROJECT_PATH"

# Create a deployment directory
mkdir -p deploy_temp
cp -r app.py templates static *.csv *.pkl requirements.txt wsgi.py Procfile runtime.txt .gitignore deploy_temp/
cp Dockerfile deploy_temp/ 2>/dev/null || true

# Create a compressed package
tar -czf deployment_package.tar.gz -C deploy_temp .

echo "📤 Uploading files to Oracle Cloud instance..."

# Upload the package to Oracle Cloud
scp deployment_package.tar.gz $ORACLE_USER@$ORACLE_IP:~/

echo "🔧 Setting up the application on Oracle Cloud..."

# Execute deployment commands on Oracle Cloud
ssh $ORACLE_USER@$ORACLE_IP << 'EOF'
    echo "🔄 Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    
    echo "🐍 Installing Python and dependencies..."
    sudo apt install python3 python3-pip python3-venv python3-dev build-essential -y
    
    echo "📁 Creating application directory..."
    sudo mkdir -p /opt/$PROJECT_NAME
    sudo chown $USER:$USER /opt/$PROJECT_NAME
    
    echo "📦 Extracting application files..."
    cd /opt/$PROJECT_NAME
    tar -xzf ~/deployment_package.tar.gz
    rm ~/deployment_package.tar.gz
    
    echo "🔧 Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "📦 Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo "🔧 Installing Gunicorn..."
    pip install gunicorn
    
    echo "⚙️ Creating systemd service..."
    sudo tee /etc/systemd/system/$PROJECT_NAME.service > /dev/null << SERVICE_EOF
[Unit]
Description=Crop Recommendation Flask App
After=network.target

[Service]
User=$USER
WorkingDirectory=/opt/$PROJECT_NAME
Environment="PATH=/opt/$PROJECT_NAME/venv/bin"
ExecStart=/opt/$PROJECT_NAME/venv/bin/gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 wsgi:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE_EOF

    echo "🔧 Configuring firewall..."
    sudo ufw allow 22
    sudo ufw allow 80
    sudo ufw allow 443
    sudo ufw allow 8000
    sudo ufw --force enable
    
    echo "🚀 Starting the application service..."
    sudo systemctl daemon-reload
    sudo systemctl enable $PROJECT_NAME.service
    sudo systemctl start $PROJECT_NAME.service
    
    echo "📊 Checking service status..."
    sudo systemctl status $PROJECT_NAME.service --no-pager
    
    echo "🌐 Application should be running on: http://$ORACLE_IP:8000"
EOF

# Clean up local files
rm -rf deploy_temp deployment_package.tar.gz

echo "✅ Deployment completed!"
echo ""
echo "🌐 Your application is now running at: http://$ORACLE_IP:8000"
echo ""
echo "📋 Useful commands:"
echo "   Check status: ssh $ORACLE_USER@$ORACLE_IP 'sudo systemctl status $PROJECT_NAME.service'"
echo "   View logs: ssh $ORACLE_USER@$ORACLE_IP 'sudo journalctl -u $PROJECT_NAME.service -f'"
echo "   Restart app: ssh $ORACLE_USER@$ORACLE_IP 'sudo systemctl restart $PROJECT_NAME.service'"
echo ""
echo "🔧 To configure a domain name, update your DNS to point to: $ORACLE_IP" 