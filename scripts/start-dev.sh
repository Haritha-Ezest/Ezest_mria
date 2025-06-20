#!/bin/bash
set -e

echo "🚀 Starting MRIA development environment..."

# Check if Redis is running (Docker or local)
if ! redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
    echo "📦 Redis not accessible. Checking if Redis container is running..."
    if docker ps | grep -q "mria-redis-dev"; then
        echo "⏳ Redis container is running, waiting for it to be ready..."
        sleep 3
    else
        echo "🐳 Starting Redis with Docker..."
        if [ -f "scripts/redis-dev.sh" ]; then
            chmod +x scripts/redis-dev.sh
            ./scripts/redis-dev.sh start
        else
            echo "📦 Starting Redis container manually..."
            docker run -d --name mria-redis-dev -p 6379:6379 redis:7-alpine
            sleep 5
        fi
    fi
    
    # Wait for Redis to be ready
    echo "⏳ Waiting for Redis to be ready..."
    for i in {1..10}; do
        if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
            echo "✅ Redis is ready!"
            break
        fi
        sleep 1
    done
    
    if ! redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
        echo "❌ Failed to connect to Redis"
        exit 1
    fi
else
    echo "✅ Redis is already running and accessible"
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create storage directories
echo "📁 Creating storage directories..."
mkdir -p storage/{uploads,processed,temp}
mkdir -p logs

# Copy local environment if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.local .env
fi

# Start the application with hot reload
echo "🎯 Starting MRIA application..."
echo "📍 Application will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Interactive API: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the application"

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env
