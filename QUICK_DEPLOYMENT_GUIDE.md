# Quick Oracle Cloud Deployment Guide

## Prerequisites ✅
- Oracle Cloud compute instance created
- SSH access to your instance
- Your instance IP address

## Step 1: Get Your Instance Information

1. **Get your Oracle Cloud instance IP address**
   - Go to Oracle Cloud Console
   - Navigate to Compute > Instances
   - Copy your instance's public IP address

2. **Ensure SSH access**
   - Make sure you can SSH to your instance
   - Test: `ssh ubuntu@YOUR_INSTANCE_IP`

## Step 2: Update Deployment Script

Edit the `deploy_to_oracle.sh` file and update these values:

```bash
ORACLE_IP="YOUR_ACTUAL_IP_HERE"  # Replace with your instance IP
ORACLE_USER="ubuntu"              # Usually "ubuntu" for Ubuntu instances
PROJECT_NAME="crop-recommendation-app"
LOCAL_PROJECT_PATH="/Users/shreya/Documents/untitled folder"
```

## Step 3: Run Deployment

```bash
# Make sure you're in your project directory
cd "/Users/shreya/Documents/untitled folder"

# Run the deployment script
./deploy_to_oracle.sh
```

## Step 4: Verify Deployment

After deployment completes:

1. **Check if the application is running:**
   ```bash
   ssh ubuntu@YOUR_INSTANCE_IP 'sudo systemctl status crop-recommendation-app.service'
   ```

2. **View application logs:**
   ```bash
   ssh ubuntu@YOUR_INSTANCE_IP 'sudo journalctl -u crop-recommendation-app.service -f'
   ```

3. **Access your application:**
   - Open browser and go to: `http://YOUR_INSTANCE_IP:8000`

## Troubleshooting

### If SSH connection fails:
1. Check your instance IP address
2. Verify security list allows SSH (port 22)
3. Ensure your SSH key is added to the instance

### If application doesn't start:
1. Check logs: `ssh ubuntu@YOUR_INSTANCE_IP 'sudo journalctl -u crop-recommendation-app.service'`
2. Verify all files were uploaded correctly
3. Check if Python dependencies installed properly

### If you can't access the website:
1. Verify firewall allows port 8000: `ssh ubuntu@YOUR_INSTANCE_IP 'sudo ufw status'`
2. Check if the service is running: `ssh ubuntu@YOUR_INSTANCE_IP 'sudo systemctl status crop-recommendation-app.service'`

## Useful Commands

```bash
# Restart the application
ssh ubuntu@YOUR_INSTANCE_IP 'sudo systemctl restart crop-recommendation-app.service'

# View real-time logs
ssh ubuntu@YOUR_INSTANCE_IP 'sudo journalctl -u crop-recommendation-app.service -f'

# Check application status
ssh ubuntu@YOUR_INSTANCE_IP 'sudo systemctl status crop-recommendation-app.service'

# Update application (re-run deployment script)
./deploy_to_oracle.sh
```

## Next Steps

1. **Set up a domain name** (optional)
   - Point your domain to your instance IP
   - Configure SSL certificate

2. **Set up monitoring** (optional)
   - Configure Oracle Cloud Monitoring
   - Set up alerts for application health

3. **Backup strategy** (recommended)
   - Regular backups of your application data
   - Version control for code changes

## Security Notes

- The application runs on port 8000
- Firewall is configured to allow HTTP/HTTPS traffic
- Application runs as a system service with auto-restart
- Consider setting up SSL/HTTPS for production use 