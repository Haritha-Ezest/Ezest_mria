#!/bin/bash

# MRIA Redis-Only Development Setup
# This script manages only Redis in Docker for local development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.redis-only.yml"
PROJECT_NAME="mria-redis-dev"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to check if docker-compose is available
check_compose() {
    if command -v docker-compose > /dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Neither docker-compose nor 'docker compose' is available."
        exit 1
    fi
}

# Start Redis services
start_redis() {
    print_header "Starting Redis for Local Development"
    check_docker
    check_compose
    
    print_status "Starting Redis and Redis Commander..."
    $COMPOSE_CMD -f $COMPOSE_FILE -p $PROJECT_NAME up -d
    
    print_status "Waiting for Redis to be ready..."
    sleep 5
    
    if test_redis_connection; then
        print_success "Redis is running and ready!"
        print_status "Redis: localhost:6379"
        print_status "Redis Commander: http://localhost:8081"
        print_status ""
        print_status "You can now start your MRIA application locally:"
        print_status "  # Using UV (recommended):"
        print_status "  ./scripts/start-dev-uv.sh"
        print_status ""
        print_status "  # Using standard setup:"
        print_status "  ./scripts/start-dev.sh"
        print_status ""
        print_status "  # Or manually:"
        print_status "  source .venv/bin/activate && uvicorn app.main:app --reload"
    else
        print_error "Redis failed to start properly!"
        show_logs
        return 1
    fi
}

# Stop Redis services
stop_redis() {
    print_header "Stopping Redis Services"
    check_compose
    
    $COMPOSE_CMD -f $COMPOSE_FILE -p $PROJECT_NAME down
    print_status "Redis services stopped."
}

# Restart Redis services
restart_redis() {
    print_header "Restarting Redis Services"
    stop_redis
    sleep 2
    start_redis
}

# Show Redis logs
show_logs() {
    check_compose
    print_status "Redis logs:"
    $COMPOSE_CMD -f $COMPOSE_FILE -p $PROJECT_NAME logs --tail=50 redis
}

# Test Redis connection
test_redis_connection() {
    if command -v redis-cli > /dev/null 2>&1; then
        if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    else
        # Try with Docker if redis-cli is not available
        if docker exec mria-redis-dev redis-cli ping > /dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    fi
}

# Show Redis status
show_status() {
    print_header "Redis Development Status"
    check_compose
    
    echo "Redis Services:"
    $COMPOSE_CMD -f $COMPOSE_FILE -p $PROJECT_NAME ps
    
    echo ""
    if test_redis_connection; then
        print_success "✅ Redis is running and accessible"
    else
        print_error "❌ Redis is not accessible"
    fi
    
    echo ""
    print_status "Connection Details:"
    print_status "  Redis URL: redis://localhost:6379"
    print_status "  Redis Commander: http://localhost:8081"
}

# Reset Redis data
reset_redis() {
    print_header "Resetting Redis Data"
    print_warning "This will delete all Redis data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        check_compose
        
        print_status "Stopping Redis services..."
        $COMPOSE_CMD -f $COMPOSE_FILE -p $PROJECT_NAME down -v
        
        print_status "Starting Redis services..."
        $COMPOSE_CMD -f $COMPOSE_FILE -p $PROJECT_NAME up -d
        
        print_success "Redis data has been reset!"
    else
        print_status "Reset cancelled."
    fi
}

# Clean up Redis resources
cleanup() {
    print_header "Cleaning Up Redis Resources"
    check_compose
    
    print_warning "This will remove all Redis containers and volumes."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Stopping and removing Redis resources..."
        $COMPOSE_CMD -f $COMPOSE_FILE -p $PROJECT_NAME down -v --remove-orphans
        
        print_status "Removing Redis images..."
        docker images | grep redis | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true
        
        print_success "Cleanup complete!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Access Redis CLI
redis_cli() {
    print_status "Accessing Redis CLI..."
    if command -v redis-cli > /dev/null 2>&1; then
        redis-cli -h localhost -p 6379
    else
        print_status "Using Redis CLI from Docker container..."
        docker exec -it mria-redis-dev redis-cli
    fi
}

# Show Redis info
redis_info() {
    print_header "Redis Information"
    
    if test_redis_connection; then
        if command -v redis-cli > /dev/null 2>&1; then
            echo "Redis Server Info:"
            redis-cli -h localhost -p 6379 info server
            echo ""
            echo "Redis Memory Info:"
            redis-cli -h localhost -p 6379 info memory
        else
            echo "Redis Server Info:"
            docker exec mria-redis-dev redis-cli info server
            echo ""
            echo "Redis Memory Info:"
            docker exec mria-redis-dev redis-cli info memory
        fi
    else
        print_error "Redis is not accessible!"
        return 1
    fi
}

# Test complete setup
test_setup() {
    print_header "Testing Redis Setup"
    
    # Start Redis if not running
    if ! test_redis_connection; then
        print_status "Starting Redis for testing..."
        start_redis
        sleep 5
    fi
    
    # Test Redis connection
    if test_redis_connection; then
        print_success "✅ Redis connection test passed!"
    else
        print_error "❌ Redis connection test failed!"
        return 1
    fi
    
    # Test Redis Commander
    if curl -f -s http://localhost:8081 > /dev/null 2>&1; then
        print_success "✅ Redis Commander is accessible!"
    else
        print_warning "⚠️  Redis Commander is not accessible (may still be starting)"
    fi
    
    # Test Redis operations
    print_status "Testing Redis operations..."
    if command -v redis-cli > /dev/null 2>&1; then
        redis-cli -h localhost -p 6379 set test_key "test_value" > /dev/null
        local value=$(redis-cli -h localhost -p 6379 get test_key)
        if [[ "$value" == "test_value" ]]; then
            print_success "✅ Redis read/write test passed!"
            redis-cli -h localhost -p 6379 del test_key > /dev/null
        else
            print_error "❌ Redis read/write test failed!"
            return 1
        fi
    else
        docker exec mria-redis-dev redis-cli set test_key "test_value" > /dev/null
        local value=$(docker exec mria-redis-dev redis-cli get test_key)
        if [[ "$value" == "test_value" ]]; then
            print_success "✅ Redis read/write test passed!"
            docker exec mria-redis-dev redis-cli del test_key > /dev/null
        else
            print_error "❌ Redis read/write test failed!"
            return 1
        fi
    fi
    
    print_success "🎉 All Redis tests passed!"
    print_status "You can now start your MRIA application locally."
}

# Help function
show_help() {
    echo "MRIA Redis-Only Development Setup"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start         - Start Redis services"
    echo "  stop          - Stop Redis services"
    echo "  restart       - Restart Redis services"
    echo "  status        - Show Redis status"
    echo "  logs          - Show Redis logs"
    echo "  reset         - Reset Redis data"
    echo "  cleanup       - Remove all Redis resources"
    echo "  cli           - Access Redis CLI"
    echo "  info          - Show Redis server information"
    echo "  test          - Test Redis setup"
    echo "  help          - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start              # Start Redis for development"
    echo "  $0 logs               # View Redis logs"
    echo "  $0 cli                # Access Redis CLI"
    echo "  $0 test               # Test complete setup"
    echo ""
    echo "After starting Redis, you can run your MRIA app locally:"
    echo "  ./scripts/start-dev-uv.sh     # Using UV (recommended)"
    echo "  ./scripts/start-dev.sh        # Using standard setup"
    echo ""
    echo "Services:"
    echo "  Redis: localhost:6379"
    echo "  Redis Commander: http://localhost:8081"
}

# Main command dispatcher
main() {
    case "${1:-help}" in
        "start") start_redis ;;
        "stop") stop_redis ;;
        "restart") restart_redis ;;
        "status") show_status ;;
        "logs") show_logs ;;
        "reset") reset_redis ;;
        "cleanup") cleanup ;;
        "cli") redis_cli ;;
        "info") redis_info ;;
        "test") test_setup ;;
        "help"|"--help"|"-h") show_help ;;
        *) print_error "Unknown command: $1"; show_help; exit 1 ;;
    esac
}

# Run main function
main "$@"
