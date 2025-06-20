#!/bin/bash

# Test script for Redis-only local development setup
# This script validates that the local development environment works correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REDIS_HOST="localhost"
REDIS_PORT="6379"
APP_HOST="localhost"
APP_PORT="8000"

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

# Function to test Redis connection
test_redis_connection() {
    print_status "Testing Redis connection..."
    
    if command -v redis-cli > /dev/null 2>&1; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
            print_success "✅ Redis connection successful"
            return 0
        else
            print_error "❌ Redis connection failed"
            return 1
        fi
    else
        print_warning "redis-cli not available, trying Docker exec..."
        if docker exec mria-redis-dev redis-cli ping > /dev/null 2>&1; then
            print_success "✅ Redis connection successful (via Docker)"
            return 0
        else
            print_error "❌ Redis connection failed"
            return 1
        fi
    fi
}

# Function to test Redis operations
test_redis_operations() {
    print_status "Testing Redis operations..."
    
    local test_key="mria_test_key"
    local test_value="test_value_$(date +%s)"
    
    if command -v redis-cli > /dev/null 2>&1; then
        # Set a test value
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" set "$test_key" "$test_value" > /dev/null
        
        # Get the value back
        local retrieved_value=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" get "$test_key")
        
        if [[ "$retrieved_value" == "$test_value" ]]; then
            print_success "✅ Redis read/write operations working"
            
            # Clean up
            redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" del "$test_key" > /dev/null
            return 0
        else
            print_error "❌ Redis read/write operations failed"
            return 1
        fi
    else
        # Try with Docker
        docker exec mria-redis-dev redis-cli set "$test_key" "$test_value" > /dev/null
        local retrieved_value=$(docker exec mria-redis-dev redis-cli get "$test_key")
        
        if [[ "$retrieved_value" == "$test_value" ]]; then
            print_success "✅ Redis read/write operations working (via Docker)"
            docker exec mria-redis-dev redis-cli del "$test_key" > /dev/null
            return 0
        else
            print_error "❌ Redis read/write operations failed"
            return 1
        fi
    fi
}

# Function to test Redis Commander
test_redis_commander() {
    print_status "Testing Redis Commander..."
    
    if curl -f -s "http://$REDIS_HOST:8081" > /dev/null 2>&1; then
        print_success "✅ Redis Commander is accessible"
        return 0
    else
        print_warning "⚠️  Redis Commander is not accessible (may not be running)"
        return 1
    fi
}

# Function to test Python environment
test_python_environment() {
    print_status "Testing Python environment..."
    
    # Check if virtual environment exists
    if [[ -d ".venv" ]]; then
        print_success "✅ Virtual environment exists"
    else
        print_error "❌ Virtual environment not found"
        return 1
    fi
    
    # Activate virtual environment and test imports
    source .venv/bin/activate
    
    # Test basic imports
    if python -c "import sys; print('Python version:', sys.version)" 2>/dev/null; then
        print_success "✅ Python environment is working"
    else
        print_error "❌ Python environment has issues"
        return 1
    fi
    
    # Test Redis client import
    if python -c "import redis; print('Redis client imported successfully')" 2>/dev/null; then
        print_success "✅ Redis client library available"
    else
        print_error "❌ Redis client library not available"
        return 1
    fi
    
    # Test application import
    if python -c "from app.main import app; print('MRIA app imported successfully')" 2>/dev/null; then
        print_success "✅ MRIA application can be imported"
    else
        print_error "❌ MRIA application import failed"
        return 1
    fi
    
    return 0
}

# Function to test application startup (background)
test_app_startup() {
    print_status "Testing application startup..."
    
    # Source environment and start app in background
    source .venv/bin/activate
    
    # Load environment variables
    if [[ -f ".env.local" ]]; then
        export $(grep -v '^#' .env.local | xargs)
    fi
    
    # Start the application in background
    uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT" > /dev/null 2>&1 &
    local app_pid=$!
    
    # Wait for application to start
    print_status "Waiting for application to start..."
    local max_attempts=15
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s "http://$APP_HOST:$APP_PORT/health" > /dev/null 2>&1; then
            print_success "✅ Application started successfully"
            
            # Kill the application
            kill $app_pid 2>/dev/null || true
            wait $app_pid 2>/dev/null || true
            
            return 0
        fi
        
        ((attempt++))
        sleep 1
    done
    
    print_error "❌ Application failed to start within timeout"
    
    # Kill the application
    kill $app_pid 2>/dev/null || true
    wait $app_pid 2>/dev/null || true
    
    return 1
}

# Function to test application endpoints
test_app_endpoints() {
    print_status "Testing application endpoints..."
    
    # Source environment and start app in background
    source .venv/bin/activate
    
    # Load environment variables
    if [[ -f ".env.local" ]]; then
        export $(grep -v '^#' .env.local | xargs)
    fi
    
    # Start the application in background
    uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT" > /dev/null 2>&1 &
    local app_pid=$!
    
    # Wait for application to start
    sleep 5
    
    # Test health endpoint
    if curl -f -s "http://$APP_HOST:$APP_PORT/health" > /dev/null 2>&1; then
        print_success "✅ Health endpoint is working"
    else
        print_error "❌ Health endpoint failed"
        kill $app_pid 2>/dev/null || true
        return 1
    fi
    
    # Test docs endpoint
    if curl -f -s "http://$APP_HOST:$APP_PORT/docs" > /dev/null 2>&1; then
        print_success "✅ Documentation endpoint is working"
    else
        print_warning "⚠️  Documentation endpoint not accessible"
    fi
    
    # Test root endpoint
    if curl -f -s "http://$APP_HOST:$APP_PORT/" > /dev/null 2>&1; then
        print_success "✅ Root endpoint is working"
    else
        print_warning "⚠️  Root endpoint not accessible"
    fi
    
    # Kill the application
    kill $app_pid 2>/dev/null || true
    wait $app_pid 2>/dev/null || true
    
    return 0
}

# Function to test storage directories
test_storage_setup() {
    print_status "Testing storage directories..."
    
    local directories=(
        "storage/uploads"
        "storage/processed"
        "storage/temp"
        "logs"
    )
    
    local all_exist=true
    for dir in "${directories[@]}"; do
        if [[ -d "$dir" ]]; then
            print_status "✅ Directory exists: $dir"
        else
            print_error "❌ Directory missing: $dir"
            all_exist=false
        fi
    done
    
    if [[ "$all_exist" == "true" ]]; then
        print_success "✅ All storage directories exist"
        return 0
    else
        print_error "❌ Some storage directories are missing"
        return 1
    fi
}

# Function to test environment configuration
test_environment_config() {
    print_status "Testing environment configuration..."
    
    if [[ -f ".env.local" ]]; then
        print_success "✅ Environment file exists"
        
        # Check key configurations
        if grep -q "REDIS_URL" .env.local; then
            print_success "✅ Redis URL configured"
        else
            print_error "❌ Redis URL not configured"
            return 1
        fi
        
        if grep -q "ENVIRONMENT=development" .env.local; then
            print_success "✅ Development environment configured"
        else
            print_warning "⚠️  Development environment not explicitly set"
        fi
        
        return 0
    else
        print_error "❌ Environment file not found"
        return 1
    fi
}

# Function to run all tests
run_all_tests() {
    print_header "MRIA Redis-Only Local Development Test Suite"
    
    local failed_tests=0
    local total_tests=0
    
    # Test Redis
    ((total_tests++))
    if ! test_redis_connection; then
        ((failed_tests++))
    fi
    
    ((total_tests++))
    if ! test_redis_operations; then
        ((failed_tests++))
    fi
    
    ((total_tests++))
    if ! test_redis_commander; then
        ((failed_tests++))
    fi
    
    # Test Python environment
    ((total_tests++))
    if ! test_python_environment; then
        ((failed_tests++))
    fi
    
    # Test environment configuration
    ((total_tests++))
    if ! test_environment_config; then
        ((failed_tests++))
    fi
    
    # Test storage setup
    ((total_tests++))
    if ! test_storage_setup; then
        ((failed_tests++))
    fi
    
    # Test application startup
    ((total_tests++))
    if ! test_app_startup; then
        ((failed_tests++))
    fi
    
    # Test application endpoints
    ((total_tests++))
    if ! test_app_endpoints; then
        ((failed_tests++))
    fi
    
    # Summary
    print_header "Test Results"
    local passed_tests=$((total_tests - failed_tests))
    
    if [[ $failed_tests -eq 0 ]]; then
        print_success "🎉 All $total_tests tests passed!"
        echo -e "\n${GREEN}Your Redis-only local development setup is working perfectly!${NC}"
        echo -e "\n${BLUE}Quick Start:${NC}"
        echo -e "  ${GREEN}1.${NC} Redis is running: ./scripts/redis-dev.sh status"
        echo -e "  ${GREEN}2.${NC} Start development: ./scripts/start-dev-uv.sh"
        echo -e "  ${GREEN}3.${NC} Access API: http://localhost:8000"
        echo -e "  ${GREEN}4.${NC} Redis Commander: http://localhost:8081"
    else
        print_error "❌ $failed_tests out of $total_tests tests failed!"
        echo -e "\n${RED}Some components need attention:${NC}"
        echo -e "  ${YELLOW}1.${NC} Check Redis: ./scripts/redis-dev.sh start"
        echo -e "  ${YELLOW}2.${NC} Check dependencies: uv pip install -r requirements.txt"
        echo -e "  ${YELLOW}3.${NC} Check environment: cp .env.local .env"
        echo -e "  ${YELLOW}4.${NC} Run setup: ./scripts/start-dev-uv.sh --test-only"
        return 1
    fi
}

# Function to show help
show_help() {
    echo "MRIA Redis-Only Local Development Test Suite"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  all          - Run all tests (default)"
    echo "  redis        - Test Redis connection and operations"
    echo "  python       - Test Python environment"
    echo "  app          - Test application startup"
    echo "  endpoints    - Test application endpoints"
    echo "  storage      - Test storage directories"
    echo "  env          - Test environment configuration"
    echo "  help         - Show this help"
    echo ""
    echo "Prerequisites:"
    echo "  - Redis running in Docker (./scripts/redis-dev.sh start)"
    echo "  - Python virtual environment set up"
    echo "  - Dependencies installed"
    echo ""
    echo "This script validates your Redis-only local development setup."
}

# Function to run individual tests
run_individual_test() {
    case "$1" in
        "redis")
            test_redis_connection && test_redis_operations && test_redis_commander
            ;;
        "python")
            test_python_environment
            ;;
        "app")
            test_app_startup
            ;;
        "endpoints")
            test_app_endpoints
            ;;
        "storage")
            test_storage_setup
            ;;
        "env")
            test_environment_config
            ;;
        *)
            print_error "Unknown test: $1"
            show_help
            exit 1
            ;;
    esac
}

# Main function
main() {
    # Ensure we're in the right directory
    if [[ ! -f "requirements.txt" ]]; then
        print_error "This script must be run from the project root directory!"
        exit 1
    fi
    
    case "${1:-all}" in
        "all")
            run_all_tests
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            run_individual_test "$1"
            ;;
    esac
}

# Run main function
main "$@"
