#!/bin/bash

# MRIA Docker Management Script with UV Support
# This script provides commands to manage MRIA Docker containers with UV optimization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_COMPOSE_FILE="docker-compose.uv.yml"
PROD_COMPOSE_FILE="docker-compose.prod.yml"
PROJECT_NAME="mria"

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

print_header() {
    echo -e "${BLUE}================================${NC}"
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

# Development mode commands
dev_start() {
    print_header "Starting MRIA Development Environment with UV"
    check_docker
    check_compose
    
    print_status "Building and starting services..."
    $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" up --build -d
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    print_status "Services started successfully!"
    print_status "MRIA API: http://localhost:8000"
    print_status "Redis Commander: http://localhost:8081"
    print_status "View logs: $0 dev logs"
}

dev_stop() {
    print_header "Stopping MRIA Development Environment"
    check_compose
    
    $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" down
    print_status "Development environment stopped."
}

dev_restart() {
    print_header "Restarting MRIA Development Environment"
    dev_stop
    sleep 2
    dev_start
}

dev_logs() {
    check_compose
    $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" logs -f "${2:-mria-app}"
}

dev_shell() {
    check_compose
    print_status "Opening shell in MRIA container..."
    $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" exec mria-app /bin/bash
}

dev_rebuild() {
    print_header "Rebuilding MRIA Development Environment"
    check_compose
    
    print_status "Stopping services..."
    $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" down
    
    print_status "Rebuilding images..."
    $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" build --no-cache
    
    print_status "Starting services..."
    $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" up -d
    
    print_status "Rebuild complete!"
}

# Production mode commands
prod_start() {
    print_header "Starting MRIA Production Environment"
    check_docker
    check_compose
    
    print_status "Building and starting production services..."
    $COMPOSE_CMD -f $PROD_COMPOSE_FILE -p "${PROJECT_NAME}-prod" up --build -d
    
    print_status "Waiting for services to be ready..."
    sleep 15
    
    print_status "Production services started successfully!"
    print_status "MRIA API: http://localhost:8000"
    print_status "Nginx: http://localhost (if configured)"
}

prod_stop() {
    print_header "Stopping MRIA Production Environment"
    check_compose
    
    $COMPOSE_CMD -f $PROD_COMPOSE_FILE -p "${PROJECT_NAME}-prod" down
    print_status "Production environment stopped."
}

prod_logs() {
    check_compose
    $COMPOSE_CMD -f $PROD_COMPOSE_FILE -p "${PROJECT_NAME}-prod" logs -f "${2:-mria-app}"
}

# Utility commands
status() {
    print_header "MRIA Docker Status"
    check_compose
    
    echo "Development Environment:"
    $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" ps 2>/dev/null || echo "Not running"
    
    echo ""
    echo "Production Environment:"
    $COMPOSE_CMD -f $PROD_COMPOSE_FILE -p "${PROJECT_NAME}-prod" ps 2>/dev/null || echo "Not running"
}

cleanup() {
    print_header "Cleaning Up MRIA Docker Resources"
    check_compose
    
    print_warning "This will remove all MRIA containers, images, and volumes."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Stopping all services..."
        $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" down -v --remove-orphans 2>/dev/null || true
        $COMPOSE_CMD -f $PROD_COMPOSE_FILE -p "${PROJECT_NAME}-prod" down -v --remove-orphans 2>/dev/null || true
        
        print_status "Removing images..."
        docker images | grep mria | awk '{print $3}' | xargs -r docker rmi -f
        
        print_status "Cleanup complete!"
    else
        print_status "Cleanup cancelled."
    fi
}

test_setup() {
    print_header "Testing MRIA Docker Setup"
    check_docker
    check_compose
    
    print_status "Testing development environment..."
    dev_start
    
    sleep 15
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "✅ Health check passed!"
    else
        print_error "❌ Health check failed!"
        dev_logs
        return 1
    fi
    
    if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
        print_status "✅ Redis connection successful!"
    else
        print_error "❌ Redis connection failed!"
        return 1
    fi
    
    print_status "🎉 All tests passed!"
    print_status "Use '$0 dev stop' to stop the test environment."
}

# UV-specific commands
uv_cache_clear() {
    print_header "Clearing UV Cache in Containers"
    check_compose
    
    print_status "Clearing UV cache..."
    $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" exec mria-app uv cache clean 2>/dev/null || true
    docker volume rm "${PROJECT_NAME}-dev_uv_cache" 2>/dev/null || true
    
    print_status "UV cache cleared!"
}

uv_update() {
    print_header "Updating Dependencies with UV"
    check_compose
    
    print_status "Updating Python dependencies..."
    $COMPOSE_CMD -f $DEFAULT_COMPOSE_FILE -p "${PROJECT_NAME}-dev" exec mria-app uv pip install -r requirements.txt --upgrade
    
    print_status "Dependencies updated!"
}

# Help function
show_help() {
    echo "MRIA Docker Management Script with UV Support"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Development Commands:"
    echo "  dev start     - Start development environment with UV optimization"
    echo "  dev stop      - Stop development environment"
    echo "  dev restart   - Restart development environment"
    echo "  dev logs      - View development logs"
    echo "  dev shell     - Open shell in development container"
    echo "  dev rebuild   - Rebuild development environment"
    echo ""
    echo "Production Commands:"
    echo "  prod start    - Start production environment"
    echo "  prod stop     - Stop production environment"
    echo "  prod logs     - View production logs"
    echo ""
    echo "Utility Commands:"
    echo "  status        - Show status of all environments"
    echo "  cleanup       - Remove all MRIA Docker resources"
    echo "  test          - Test the Docker setup"
    echo ""
    echo "UV Commands:"
    echo "  uv-cache-clear - Clear UV cache in containers"
    echo "  uv-update      - Update dependencies with UV"
    echo ""
    echo "Examples:"
    echo "  $0 dev start          # Start development environment"
    echo "  $0 dev logs mria-app  # View app logs"
    echo "  $0 prod start         # Start production environment"
    echo "  $0 status             # Check all environments"
}

# Main command dispatcher
main() {
    case "${1:-help}" in
        "dev")
            case "${2:-start}" in
                "start") dev_start ;;
                "stop") dev_stop ;;
                "restart") dev_restart ;;
                "logs") dev_logs "$@" ;;
                "shell") dev_shell ;;
                "rebuild") dev_rebuild ;;
                *) print_error "Unknown dev command: $2"; show_help; exit 1 ;;
            esac
            ;;
        "prod")
            case "${2:-start}" in
                "start") prod_start ;;
                "stop") prod_stop ;;
                "logs") prod_logs "$@" ;;
                *) print_error "Unknown prod command: $2"; show_help; exit 1 ;;
            esac
            ;;
        "status") status ;;
        "cleanup") cleanup ;;
        "test") test_setup ;;
        "uv-cache-clear") uv_cache_clear ;;
        "uv-update") uv_update ;;
        "help"|"--help"|"-h") show_help ;;
        *) print_error "Unknown command: $1"; show_help; exit 1 ;;
    esac
}

# Run main function
main "$@"
