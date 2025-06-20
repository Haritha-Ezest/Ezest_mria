#!/bin/bash
set -e

echo "🐳 Starting MRIA with Docker Compose..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose."
    exit 1
fi

# Create necessary directories
echo "📁 Creating storage directories..."
mkdir -p storage/{uploads,processed,temp}
mkdir -p logs

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service status
echo "📊 Checking service status..."
docker-compose ps

# Test if services are responding
echo "🧪 Testing service health..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ MRIA application is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Application failed to start within 30 seconds"
        docker-compose logs
        exit 1
    fi
    sleep 1
done

echo ""
echo "🎉 MRIA Docker environment is ready!"
echo "📍 Application: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔧 Redis: localhost:6379"
echo ""
echo "📋 Useful commands:"
echo "  docker-compose logs -f          # View logs"
echo "  docker-compose stop             # Stop services"
echo "  docker-compose down             # Stop and remove containers"
echo "  scripts/reset-redis.sh          # Reset Redis data"
