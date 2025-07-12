# Oracle Cloud Deployment Guide

## Prerequisites
1. Oracle Cloud account
2. Oracle Cloud CLI installed
3. Docker (for container deployment)

## Method 1: Container Deployment (Recommended)

### Step 1: Prepare Your Application
Your application is already prepared with:
- ✅ `requirements.txt` - Dependencies
- ✅ `Dockerfile` - Container configuration
- ✅ `wsgi.py` - WSGI entry point
- ✅ `Procfile` - Process configuration

### Step 2: Build and Push Docker Image

```bash
# Build the Docker image
docker build -t crop-recommendation-app .

# Tag the image for Oracle Container Registry
docker tag crop-recommendation-app:latest <region>.ocir.io/<tenancy-namespace>/<repo-name>/crop-recommendation-app:latest

# Login to Oracle Container Registry
docker login <region>.ocir.io

# Push the image
docker push <region>.ocir.io/<tenancy-namespace>/<repo-name>/crop-recommendation-app:latest
```

### Step 3: Deploy to Oracle Container Engine for Kubernetes (OKE)

1. **Create Kubernetes Cluster** (if not exists):
   - Go to Oracle Cloud Console
   - Navigate to Developer Services > Container Engine for Kubernetes
   - Create a new cluster

2. **Create Deployment YAML**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crop-recommendation-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: crop-recommendation-app
  template:
    metadata:
      labels:
        app: crop-recommendation-app
    spec:
      containers:
      - name: crop-recommendation-app
        image: <region>.ocir.io/<tenancy-namespace>/<repo-name>/crop-recommendation-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: FLASK_ENV
          value: "production"
```

3. **Create Service**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: crop-recommendation-service
spec:
  selector:
    app: crop-recommendation-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Method 2: Oracle Cloud Infrastructure (OCI) Compute Instance

### Step 1: Create Compute Instance
1. Go to Oracle Cloud Console
2. Navigate to Compute > Instances
3. Create a new instance with Ubuntu 22.04

### Step 2: Configure the Instance
```bash
# SSH into your instance
ssh ubuntu@<your-instance-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv -y

# Install Git
sudo apt install git -y

# Clone your repository
git clone <your-repo-url>
cd <your-project-directory>

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Gunicorn
pip install gunicorn
```

### Step 3: Configure Application
```bash
# Create systemd service
sudo nano /etc/systemd/system/crop-recommendation.service
```

Add this content:
```ini
[Unit]
Description=Crop Recommendation Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/<your-project-directory>
Environment="PATH=/home/ubuntu/<your-project-directory>/venv/bin"
ExecStart=/home/ubuntu/<your-project-directory>/venv/bin/gunicorn --bind 0.0.0.0:8000 wsgi:app
Restart=always

[Install]
WantedBy=multi-user.target
```

### Step 4: Start the Service
```bash
# Enable and start the service
sudo systemctl enable crop-recommendation.service
sudo systemctl start crop-recommendation.service

# Check status
sudo systemctl status crop-recommendation.service
```

### Step 5: Configure Firewall
```bash
# Allow HTTP traffic
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 8000
```

## Method 3: Oracle Functions (Serverless)

### Step 1: Install Fn CLI
```bash
curl -LSs https://raw.githubusercontent.com/fnproject/cli/master/install | sh
```

### Step 2: Create Function
```bash
# Initialize function
fn init --runtime python crop-recommendation-func

# Deploy function
fn deploy --app crop-recommendation-app
```

## Environment Variables
Set these in your deployment:
- `FLASK_ENV=production`
- `FLASK_APP=app.py`

## Monitoring and Logs
- Use Oracle Cloud Monitoring for metrics
- View logs in Oracle Cloud Console
- Set up alerts for application health

## SSL/HTTPS Configuration
1. Obtain SSL certificate (Let's Encrypt or Oracle SSL)
2. Configure load balancer with SSL termination
3. Update security lists to allow HTTPS traffic

## Backup Strategy
1. Regular database backups (if applicable)
2. Application code version control
3. Configuration backups
4. Disaster recovery plan

## Cost Optimization
1. Use appropriate instance shapes
2. Enable auto-scaling
3. Monitor resource usage
4. Use reserved instances for predictable workloads

## Security Best Practices
1. Use VCN (Virtual Cloud Network) with proper security lists
2. Implement least privilege access
3. Regular security updates
4. Monitor access logs
5. Use Oracle Cloud Guard for security monitoring

## Troubleshooting
- Check application logs: `sudo journalctl -u crop-recommendation.service`
- Verify network connectivity
- Check resource usage
- Review security list configurations 