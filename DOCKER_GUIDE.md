# 🐳 Docker Containerization Guide

Complete guide for containerizing and deploying the AI RAG Research Assistant using Docker.

## 📋 Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose
- Git
- At least 4GB RAM available for Docker

### Install Docker

**Windows/Mac:**

- Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
- Follow installation instructions
- Verify installation: `docker --version`

**Linux (Ubuntu/Debian):**

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/ali-mansha1351/AI_RAG_ResearchAssistant.git
cd AI_RAG_ResearchAssistant

# Copy environment template
cp .env.example .env
```

### 2. Configure Environment

Edit the `.env` file with your API keys:

```env
# Required: Add your API keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Adjust performance settings
MAX_RESULTS=5
BATCH_SIZE=32
SIMILARITY_THRESHOLD=0.7
USE_LOCAL_EMBEDDINGS=true
```

### 3. Build and Run

```bash
# Build and start the application
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 4. Access Application

- Open your browser to `http://localhost:8501`
- Upload documents and start chatting!

## 📁 File Structure

After setup, your project should look like:

```
AI_RAG_ResearchAssistant/
├── 🐳 Docker Files
│   ├── Dockerfile              # Main container definition
│   ├── docker-compose.yml      # Multi-container orchestration
│   ├── .dockerignore           # Files to exclude from build
│   ├── nginx.conf              # Reverse proxy configuration
│   └── requirements.txt        # Python dependencies
├── 🔧 Configuration
│   ├── .env                    # Environment variables (create from .env.example)
│   ├── .env.example            # Environment template
│   └── config/settings.py      # Application settings
├── 📂 Application Code
│   ├── streamlit_app.py        # Main web application
│   ├── src/                    # Source code modules
│   └── data/                   # Persistent data (auto-created)
└── 📖 Documentation
    ├── README.md               # Main documentation
    └── DOCKER_GUIDE.md         # This guide
```

## 🔧 Docker Configuration

### Dockerfile Explained

```dockerfile
# Use Python 3.11 slim for smaller image size
FROM python:3.11-slim

# Environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential curl software-properties-common git

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/vector_store logs

# Expose port and add health check
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose Services

**Main Application:**

- Runs the Streamlit app on port 8501
- Mounts volumes for persistent data
- Includes health checks and restart policies

**Nginx Reverse Proxy (Optional):**

- Provides load balancing and SSL termination
- Rate limiting and security headers
- Only runs with `--profile production`

## 🛠️ Development Workflow

### Building and Testing

```bash
# Build the image
docker build -t ai-rag-assistant .

# Test run with specific settings
docker run -p 8501:8501 \
  -e GROQ_API_KEY="your_key" \
  -v $(pwd)/data:/app/data \
  ai-rag-assistant

# Check container health
docker ps
curl http://localhost:8501/_stcore/health
```

### Making Changes

```bash
# After making code changes, rebuild
docker-compose build

# Restart with new code
docker-compose up -d

# View logs
docker-compose logs -f ai-rag-assistant
```

### Data Management

```bash
# Backup data
docker run --rm -v airagproject_ai_rag_data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .

# Restore data
docker run --rm -v airagproject_ai_rag_data:/data -v $(pwd):/backup alpine tar xzf /backup/data-backup.tar.gz -C /data
```

## 🌐 Production Deployment

### 1. Basic Production Setup

```bash
# Use production profile with Nginx
docker-compose --profile production up -d

# Check services
docker-compose ps
```

### 2. SSL/HTTPS Setup

Create SSL certificates:

```bash
# Create SSL directory
mkdir ssl

# Generate self-signed certificates (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem

# Or copy your real certificates
cp your-cert.pem ssl/cert.pem
cp your-key.pem ssl/key.pem
```

Update `nginx.conf`:

```bash
# Uncomment HTTPS server block in nginx.conf
# Update server_name with your domain
```

### 3. Environment Security

```bash
# Set secure environment variables
export GROQ_API_KEY="your_secure_key"
export OPENAI_API_KEY="your_secure_key"

# Use Docker secrets (recommended)
echo "your_groq_key" | docker secret create groq_api_key -
echo "your_openai_key" | docker secret create openai_api_key -
```

### 4. Resource Limits

Update `docker-compose.yml`:

```yaml
services:
  ai-rag-assistant:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"
```

## 📊 Monitoring and Logging

### Container Monitoring

```bash
# View resource usage
docker stats

# Container health status
docker-compose ps

# Service logs
docker-compose logs -f ai-rag-assistant
docker-compose logs -f nginx

# Follow specific container logs
docker logs -f container_name
```

### Application Monitoring

```bash
# Health check endpoint
curl http://localhost:8501/_stcore/health

# Application metrics (if available)
curl http://localhost:8501/_stcore/metrics

# Check vector store status
docker-compose exec ai-rag-assistant ls -la data/vector_store/
```

### Log Management

```bash
# View application logs
docker-compose logs --tail=100 ai-rag-assistant

# Save logs to file
docker-compose logs ai-rag-assistant > app.log

# Rotate logs (add to crontab)
docker-compose logs --tail=1000 ai-rag-assistant > logs/app-$(date +%Y%m%d).log
```

## 🔄 Scaling and High Availability

### Horizontal Scaling

```bash
# Run multiple instances
docker-compose up --scale ai-rag-assistant=3

# With production setup
docker-compose --profile production up --scale ai-rag-assistant=3 -d
```

### Load Balancing

Update `nginx.conf` for multiple instances:

```nginx
upstream streamlit {
    server ai-rag-assistant_1:8501;
    server ai-rag-assistant_2:8501;
    server ai-rag-assistant_3:8501;
}
```

### Data Persistence

```bash
# Use named volumes for better performance
docker volume create ai_rag_data
docker volume create ai_rag_logs

# Backup strategy
docker run --rm -v ai_rag_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/backup-$(date +%Y%m%d).tar.gz -C /data .
```

## 🐛 Troubleshooting

### Common Issues

**Container fails to start:**

```bash
# Check Docker daemon
docker info

# Check port conflicts
netstat -tulpn | grep 8501

# View detailed logs
docker-compose logs ai-rag-assistant
```

**Permission denied errors:**

```bash
# Fix file permissions
sudo chown -R $(id -u):$(id -g) data/ logs/

# For SELinux systems
sudo setsebool -P httpd_can_network_connect 1
```

**Out of memory:**

```bash
# Check Docker memory allocation
docker system df

# Increase Docker memory (Docker Desktop)
# Settings → Resources → Memory → Increase limit

# Clean up unused containers/images
docker system prune -a
```

**Network connectivity:**

```bash
# Check Docker networks
docker network ls
docker network inspect airagproject_ai-rag-network

# Test container connectivity
docker-compose exec ai-rag-assistant ping google.com
```

### Performance Optimization

**Container performance:**

- Use local embeddings instead of API calls
- Mount `/tmp` as tmpfs for temporary files
- Use multi-stage builds for smaller images
- Enable BuildKit for faster builds

**Application tuning:**

```env
# In .env file
BATCH_SIZE=16          # Reduce for less memory usage
USE_LOCAL_EMBEDDINGS=true
MAX_RESULTS=3          # Reduce retrieval count
```

## 🔒 Security Best Practices

### Container Security

1. **Use non-root user:**

```dockerfile
RUN adduser --disabled-password --gecos '' appuser
USER appuser
```

2. **Scan for vulnerabilities:**

```bash
docker scan ai-rag-assistant
```

3. **Use secrets management:**

```bash
echo "api_key" | docker secret create groq_key -
```

### Network Security

1. **Use custom networks:**

```yaml
networks:
  ai-rag-network:
    driver: bridge
    internal: true
```

2. **Configure firewall:**

```bash
# Allow only necessary ports
sudo ufw allow 80
sudo ufw allow 443
sudo ufw deny 8501  # Hide direct access
```

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Streamlit Docker Deployment](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Nginx Docker Guide](https://hub.docker.com/_/nginx)

## 🆘 Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review Docker logs: `docker-compose logs`
3. Open an issue on [GitHub](https://github.com/ali-mansha1351/AI_RAG_ResearchAssistant/issues)
4. Join the discussions for community support

---

**Happy Containerizing! 🐳**
