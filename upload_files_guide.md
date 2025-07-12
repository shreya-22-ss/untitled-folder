# How to Upload Your Application Files

## Option 1: Using Git (Recommended)

1. **Create a GitHub repository** (if you don't have one)
2. **Upload your project files to GitHub**
3. **On your Oracle Cloud instance, run:**
   ```bash
   cd /opt/crop-recommendation-app
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git .
   ```

## Option 2: Manual File Upload

### Step 1: Create the app.py file on your instance
```bash
cd /opt/crop-recommendation-app
nano app.py
```

### Step 2: Copy and paste your app.py content
Copy the content from your local `app.py` file and paste it into the nano editor.

### Step 3: Create templates directory
```bash
mkdir -p templates
mkdir -p static
```

### Step 4: Upload template files
You'll need to manually create each template file. For example:
```bash
nano templates/index.html
# Copy and paste your index.html content
```

## Option 3: Using SCP (if you can SSH from your local machine)

If you can SSH to your instance, you can upload files using SCP:
```bash
scp -r /Users/shreya/Documents/untitled\ folder/* ubuntu@150.136.76.172:/opt/crop-recommendation-app/
```

## Files You Need to Upload:

1. **app.py** - Main Flask application
2. **templates/** - All HTML template files
3. **static/** - CSS, JS, and image files
4. ***.csv** - Data files
5. ***.pkl** - Model files

## Quick Test

After uploading files, test your application:
```bash
cd /opt/crop-recommendation-app
source venv/bin/activate
python app.py
```

Your application should be accessible at: http://150.136.76.172:8000 