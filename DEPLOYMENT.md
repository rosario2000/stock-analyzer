# 🚀 Deployment Guide - Advanced Stock Analyzer

This guide covers deploying your Advanced Stock Analyzer to various cloud platforms.

## 📁 Project Files

- `app.py` - Main application file (optimized for deployment)
- `stock_analyzer.py` - Core analysis engine with LSTM
- `requirements.txt` - Python dependencies
- `Procfile` - Heroku configuration
- `runtime.txt` - Python version specification
- `.streamlit/config.toml` - Streamlit configuration

## 🌟 Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)

**Pros:** Free, easy setup, built for Streamlit apps
**Cons:** May have limitations with heavy ML models

#### Steps:
1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configure Secrets (if needed):**
   - In Streamlit Cloud dashboard, go to app settings
   - Add any API keys in the "Secrets" section

---

### Option 2: Railway (Modern & Fast)

**Pros:** Modern platform, good for Python apps, free tier
**Cons:** Limited free tier

#### Steps:
1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy:**
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Set Environment Variables:**
   ```bash
   railway variables set STREAMLIT_SERVER_PORT=8080
   ```

---

### Option 3: Render (Reliable)

**Pros:** Reliable, good free tier, automatic deployments
**Cons:** Cold starts on free tier

#### Steps:
1. **Create `render.yaml`:**
   ```yaml
   services:
     - type: web
       name: stock-analyzer
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy:**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Select "Web Service"
   - Configure build and start commands

---

### Option 4: Heroku (Traditional)

**Pros:** Mature platform, lots of documentation
**Cons:** No free tier anymore

#### Steps:
1. **Install Heroku CLI:**
   ```bash
   # On macOS
   brew install heroku/brew/heroku
   ```

2. **Create Heroku App:**
   ```bash
   heroku create your-stock-analyzer
   heroku buildpacks:set heroku/python
   ```

3. **Deploy:**
   ```bash
   git push heroku main
   ```

4. **Scale and Open:**
   ```bash
   heroku ps:scale web=1
   heroku open
   ```

---

## ⚙️ Environment Configuration

### For Heavy ML Workloads (LSTM):

If your deployment platform struggles with TensorFlow, you can:

1. **Create a lightweight version:**
   ```python
   # In app.py, modify the analyzer initialization
   analyzer = StockAnalyzer(use_lstm=False)  # Statistical only
   ```

2. **Use CPU-optimized TensorFlow:**
   ```bash
   # Replace in requirements.txt
   tensorflow-cpu>=2.13.0
   ```

3. **Optimize memory usage:**
   ```python
   # Add to app.py
   import os
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   ```

---

## 🔧 Troubleshooting

### Common Issues:

#### 1. **Memory Issues:**
```bash
# Add to requirements.txt
tensorflow-cpu>=2.13.0  # Instead of tensorflow
```

#### 2. **Build Timeout:**
```bash
# Simplify requirements.txt to essentials only
streamlit>=1.28.0
requests>=2.31.0
pandas>=2.2.0
numpy>=1.26.0
plotly>=5.17.0
```

#### 3. **Port Issues:**
```python
# In app.py, add at the top
import os
port = int(os.environ.get("PORT", 8501))
```

#### 4. **SSL Certificate Issues:**
```python
# Add to stock_analyzer.py
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

---

## 📊 Performance Optimization

### 1. **Caching:**
```python
@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_stock_data(symbol):
    # Your fetch logic here
    pass
```

### 2. **Lazy Loading:**
```python
@st.cache_resource
def load_lstm_model():
    # Only load when needed
    pass
```

### 3. **Reduce Dependencies:**
```txt
# Minimal requirements.txt for faster deployment
streamlit>=1.28.0
requests>=2.31.0
pandas>=2.2.0
numpy>=1.26.0
plotly>=5.17.0
```

---

## 🌐 Custom Domain (Optional)

### For Streamlit Cloud:
1. Go to app settings
2. Configure custom domain
3. Update DNS settings

### For Other Platforms:
Follow platform-specific domain configuration guides.

---

## 🔒 Security Considerations

1. **API Keys:** Use environment variables, never commit keys
2. **HTTPS:** Most platforms provide SSL certificates automatically
3. **Rate Limiting:** Consider implementing request limiting for API calls

---

## 📱 Mobile Optimization

The app is already responsive, but you can test on:
- **Chrome DevTools:** Mobile simulation
- **Real devices:** Test on actual mobile devices
- **Different screen sizes:** Tablet, desktop, etc.

---

## 🚀 Quick Start Commands

```bash
# 1. Prepare for deployment
git add .
git commit -m "Ready for deployment"
git push origin main

# 2. Test locally first
streamlit run app.py

# 3. Deploy to Streamlit Cloud (easiest)
# Just connect your GitHub repo at share.streamlit.io

# 4. Deploy to Railway (modern)
railway login && railway init && railway up

# 5. Deploy to Render
# Connect repo at render.com and configure
```

---

## 📞 Support

If you encounter issues:
1. Check the platform-specific logs
2. Verify all dependencies are in requirements.txt  
3. Test locally first with `streamlit run app.py`
4. Check GitHub Issues for common problems

**Your Advanced Stock Analyzer is ready for deployment! 🎉** 