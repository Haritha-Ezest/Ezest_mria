#!/bin/bash

# MRIA Local Development Startup Script with UV
# This script starts the MRIA application locally with Redis running in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_FILE=".env.local"
VENV_PATH=".venv"
REDIS_SCRIPT="scripts/redis-dev.sh"

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

# Function to check if UV is available
check_uv() {
    if ! command -v uv > /dev/null 2>&1; then
        print_error "UV is not installed. Please install it first:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  # or"
        echo "  pip install uv"
        exit 1
    fi
}

# Function to check if virtual environment exists
check_venv() {
    if [[ ! -d "$VENV_PATH" ]]; then
        print_warning "Virtual environment not found. Creating one..."
        uv venv $VENV_PATH --python 3.11
        print_success "Virtual environment created!"
    fi
}

# Function to activate virtual environment
activate_venv() {
    if [[ -f "$VENV_PATH/bin/activate" ]]; then
        source "$VENV_PATH/bin/activate"
        print_status "Virtual environment activated"
    else
        print_error "Cannot find virtual environment activation script!"
        exit 1
    fi
}

# Function to install/update dependencies
install_dependencies() {
    print_status "Installing/updating dependencies with UV..."
    uv pip install -r requirements.txt
    print_success "Dependencies installed successfully!"
}

# Function to check Redis connection
check_redis() {
    print_status "Checking Redis connection..."
    
    if command -v redis-cli > /dev/null 2>&1; then
        if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
            print_success "Redis is running and accessible!"
            return 0
        else
            print_warning "Redis is not accessible. Starting Redis..."
            return 1
        fi
    else
        print_warning "redis-cli not found. Checking if Redis container is running..."
        if docker ps | grep -q "mria-redis-dev"; then
            print_success "Redis container is running!"
            return 0
        else
            print_warning "Redis container is not running. Starting Redis..."
            return 1
        fi
    fi
}

# Function to start Redis if needed
ensure_redis() {
    if ! check_redis; then
        if [[ -f "$REDIS_SCRIPT" ]]; then
            print_status "Starting Redis with Docker..."
            chmod +x "$REDIS_SCRIPT"
            "$REDIS_SCRIPT" start
        else
            print_error "Redis script not found at $REDIS_SCRIPT"
            print_status "Please start Redis manually:"
            print_status "  docker run -d --name mria-redis-dev -p 6379:6379 redis:7-alpine"
            exit 1
        fi
    fi
}

# Function to check if environment file exists
check_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        print_warning "Environment file $ENV_FILE not found."
        print_status "Creating default environment file..."
        
        cat > "$ENV_FILE" << EOF
# MRIA Local Development Configuration
REDIS_URL=redis://localhost:6379
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
PYTHONPATH=.
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
EOF
        print_success "Default environment file created!"
    fi
}

# Function to load environment variables
load_env() {
    if [[ -f "$ENV_FILE" ]]; then
        print_status "Loading environment variables from $ENV_FILE..."
        export $(grep -v '^#' "$ENV_FILE" | xargs)
        print_success "Environment variables loaded!"
    fi
}

# Function to create storage directories
setup_storage() {
    print_status "Setting up storage directories..."
    
    directories=(
        "storage/uploads"
        "storage/uploads/documents"
        "storage/uploads/images"
        "storage/uploads/backup"
        "storage/processed"
        "storage/temp"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    print_success "Storage directories ready!"
}

# Function to run pre-flight checks
preflight_checks() {
    print_header "Pre-flight Checks"
    
    # Check if we're in the right directory
    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt not found. Are you in the project root?"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Python version: $python_version"
    
    # Check UV version
    uv_version=$(uv --version 2>&1 | awk '{print $2}')
    print_status "UV version: $uv_version"
    
    print_success "Pre-flight checks completed!"
}

# Function to test application startup
test_startup() {
    print_status "Testing application startup..."
    
    # Try to import the main module
    if python -c "from app.main import app; print('✅ App import successful')" 2>/dev/null; then
        print_success "Application import test passed!"
    else
        print_error "Application import test failed!"
        print_status "There might be missing dependencies or configuration issues."
        return 1
    fi
}

# Function to start the development server
start_server() {
    print_header "Starting MRIA Development Server"
    
    local host=${HOST:-0.0.0.0}
    local port=${PORT:-8000}
    
    print_status "Starting server at http://$host:$port"
    print_status "API Documentation: http://$host:$port/docs"
    print_status "Alternative docs: http://$host:$port/redoc"
    print_status ""
    print_status "Press Ctrl+C to stop the server"
    print_status ""
    
    # Start the server with UV
    uv run uvicorn app.main:app \
        --host "$host" \
        --port "$port" \
        --reload \
        --reload-dir app \
        --log-level info
}

# Function to show help
show_help() {
    echo "MRIA Local Development Startup Script with UV"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --no-redis        Don't start Redis automatically"
    echo "  --no-deps         Skip dependency installation"
    echo "  --test-only       Only run tests, don't start server"
    echo "  --help, -h        Show this help"
    echo ""
    echo "This script will:"
    echo "  1. Check UV installation"
    echo "  2. Create/activate virtual environment"
    echo "  3. Install/update dependencies"
    echo "  4. Start Redis in Docker (if needed)"
    echo "  5. Set up storage directories"
    echo "  6. Load environment variables"
    echo "  7. Start the MRIA development server"
    echo ""
    echo "Prerequisites:"
    echo "  - UV installed (https://github.com/astral-sh/uv)"
    echo "  - Docker running (for Redis)"
    echo "  - Python 3.11+"
    echo ""
    echo "After starting:"
    echo "  - API: http://localhost:8000"
    echo "  - Docs: http://localhost:8000/docs"
    echo "  - Redis: localhost:6379"
    echo "  - Redis Commander: http://localhost:8081"
}

# Main function
main() {
    local skip_redis=false
    local skip_deps=false
    local test_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-redis)
                skip_redis=true
                shift
                ;;
            --no-deps)
                skip_deps=true
                shift
                ;;
            --test-only)
                test_only=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    print_header "MRIA Local Development Setup with UV"
    
    # Run checks and setup
    check_uv
    preflight_checks
    check_venv
    activate_venv
    
    if [[ "$skip_deps" != "true" ]]; then
        install_dependencies
    fi
    
    check_env_file
    load_env
    setup_storage
    
    if [[ "$skip_redis" != "true" ]]; then
        ensure_redis
    fi
    
    # Test the application
    if ! test_startup; then
        print_error "Application startup test failed!"
        exit 1
    fi
    
    if [[ "$test_only" == "true" ]]; then
        print_success "All tests passed! Your development environment is ready."
        print_status "To start the server, run: $0"
        exit 0
    fi
    
    # Start the development server
    start_server
}

# Ensure we're in the right directory
if [[ ! -f "requirements.txt" ]]; then
    print_error "This script must be run from the project root directory!"
    exit 1
fi

# Run main function
main "$@"
