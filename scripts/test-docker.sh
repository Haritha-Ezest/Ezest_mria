#!/bin/bash

# MRIA Docker Test Suite
# Comprehensive testing for all Docker configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_TIMEOUT=120
HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_DELAY=5

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

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=${3:-$HEALTH_CHECK_RETRIES}
    local delay=${4:-$HEALTH_CHECK_DELAY}
    
    print_status "Waiting for $service_name to be ready..."
    
    for i in $(seq 1 $max_attempts); do
        if curl -f -s "$url" > /dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        print_status "Attempt $i/$max_attempts: $service_name not ready yet, waiting ${delay}s..."
        sleep $delay
    done
    
    print_error "$service_name failed to become ready after $max_attempts attempts"
    return 1
}

# Function to test Redis connection
test_redis() {
    local host=${1:-localhost}
    local port=${2:-6379}
    
    print_status "Testing Redis connection at $host:$port..."
    
    if command -v redis-cli > /dev/null 2>&1; then
        if redis-cli -h "$host" -p "$port" ping > /dev/null 2>&1; then
            print_success "Redis connection successful!"
            return 0
        else
            print_error "Redis connection failed!"
            return 1
        fi
    else
        print_warning "redis-cli not available, skipping Redis test"
        return 0
    fi
}

# Function to test API endpoints
test_api_endpoints() {
    local base_url=${1:-http://localhost:8000}
    
    print_status "Testing API endpoints at $base_url..."
    
    # Test health endpoint
    if curl -f -s "$base_url/health" > /dev/null 2>&1; then
        print_success "Health endpoint responding!"
    else
        print_error "Health endpoint failed!"
        return 1
    fi
    
    # Test docs endpoint
    if curl -f -s "$base_url/docs" > /dev/null 2>&1; then
        print_success "Documentation endpoint responding!"
    else
        print_warning "Documentation endpoint not responding (may be expected)"
    fi
    
    # Test basic API structure
    local response=$(curl -s "$base_url/health" 2>/dev/null || echo "failed")
    if [[ "$response" != "failed" ]]; then
        print_success "API is returning responses!"
    else
        print_error "API is not responding properly!"
        return 1
    fi
    
    return 0
}

# Function to test Docker environment
test_docker_environment() {
    local compose_file=$1
    local project_name=$2
    local test_name=$3
    
    print_header "Testing $test_name Environment"
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running!"
        return 1
    fi
    
    # Determine compose command
    if command -v docker-compose > /dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Neither docker-compose nor 'docker compose' is available!"
        return 1
    fi
    
    print_status "Using compose command: $COMPOSE_CMD"
    
    # Clean up any existing containers
    print_status "Cleaning up existing containers..."
    $COMPOSE_CMD -f "$compose_file" -p "$project_name" down -v --remove-orphans > /dev/null 2>&1 || true
    
    # Start services
    print_status "Starting services with $compose_file..."
    if ! $COMPOSE_CMD -f "$compose_file" -p "$project_name" up -d --build; then
        print_error "Failed to start services!"
        return 1
    fi
    
    # Wait for services to be ready
    print_status "Waiting for services to initialize..."
    sleep 15
    
    # Test Redis
    if ! test_redis; then
        print_error "Redis test failed!"
        $COMPOSE_CMD -f "$compose_file" -p "$project_name" logs
        $COMPOSE_CMD -f "$compose_file" -p "$project_name" down -v
        return 1
    fi
    
    # Test API
    if ! wait_for_service "http://localhost:8000/health" "MRIA API"; then
        print_error "API health check failed!"
        $COMPOSE_CMD -f "$compose_file" -p "$project_name" logs
        $COMPOSE_CMD -f "$compose_file" -p "$project_name" down -v
        return 1
    fi
    
    # Test API endpoints
    if ! test_api_endpoints; then
        print_error "API endpoint tests failed!"
        $COMPOSE_CMD -f "$compose_file" -p "$project_name" logs
        $COMPOSE_CMD -f "$compose_file" -p "$project_name" down -v
        return 1
    fi
    
    # Test container health
    print_status "Checking container health..."
    local unhealthy_containers=$($COMPOSE_CMD -f "$compose_file" -p "$project_name" ps --filter "health=unhealthy" -q)
    if [[ -n "$unhealthy_containers" ]]; then
        print_error "Some containers are unhealthy!"
        $COMPOSE_CMD -f "$compose_file" -p "$project_name" ps
        $COMPOSE_CMD -f "$compose_file" -p "$project_name" down -v
        return 1
    fi
    
    print_success "$test_name environment test completed successfully!"
    
    # Show running services
    print_status "Running services:"
    $COMPOSE_CMD -f "$compose_file" -p "$project_name" ps
    
    # Clean up
    print_status "Cleaning up test environment..."
    $COMPOSE_CMD -f "$compose_file" -p "$project_name" down -v > /dev/null 2>&1
    
    return 0
}

# Test standard Docker setup
test_standard_docker() {
    if [[ -f "docker-compose.yml" ]]; then
        test_docker_environment "docker-compose.yml" "mria-standard-test" "Standard Docker"
    else
        print_warning "docker-compose.yml not found, skipping standard Docker test"
        return 0
    fi
}

# Test UV Docker setup
test_uv_docker() {
    if [[ -f "docker-compose.uv.yml" ]]; then
        test_docker_environment "docker-compose.uv.yml" "mria-uv-test" "UV Docker"
    else
        print_warning "docker-compose.uv.yml not found, skipping UV Docker test"
        return 0
    fi
}

# Test production Docker setup
test_production_docker() {
    if [[ -f "docker-compose.prod.yml" ]]; then
        test_docker_environment "docker-compose.prod.yml" "mria-prod-test" "Production Docker"
    else
        print_warning "docker-compose.prod.yml not found, skipping production Docker test"
        return 0
    fi
}

# Test UV performance
test_uv_performance() {
    print_header "UV Performance Test"
    
    if [[ ! -f "docker-compose.uv.yml" ]]; then
        print_warning "docker-compose.uv.yml not found, skipping UV performance test"
        return 0
    fi
    
    # Determine compose command
    if command -v docker-compose > /dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Neither docker-compose nor 'docker compose' is available!"
        return 1
    fi
    
    # Clean up
    $COMPOSE_CMD -f "docker-compose.uv.yml" -p "mria-perf-test" down -v --remove-orphans > /dev/null 2>&1 || true
    
    # Build without cache and time it
    print_status "Testing UV build performance (no cache)..."
    local start_time=$(date +%s)
    
    if $COMPOSE_CMD -f "docker-compose.uv.yml" -p "mria-perf-test" build --no-cache > /dev/null 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "UV build completed in ${duration} seconds"
    else
        print_error "UV build failed!"
        return 1
    fi
    
    # Test with cache
    print_status "Testing UV build performance (with cache)..."
    start_time=$(date +%s)
    
    if $COMPOSE_CMD -f "docker-compose.uv.yml" -p "mria-perf-test" build > /dev/null 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_success "UV cached build completed in ${duration} seconds"
    else
        print_error "UV cached build failed!"
        return 1
    fi
    
    # Clean up
    $COMPOSE_CMD -f "docker-compose.uv.yml" -p "mria-perf-test" down -v --remove-orphans > /dev/null 2>&1 || true
    
    return 0
}

# Test Docker script functionality
test_docker_script() {
    print_header "Docker Script Test"
    
    if [[ ! -f "scripts/docker-uv.sh" ]]; then
        print_warning "scripts/docker-uv.sh not found, skipping script test"
        return 0
    fi
    
    local script="scripts/docker-uv.sh"
    
    # Test script help
    print_status "Testing script help..."
    if "$script" help > /dev/null 2>&1; then
        print_success "Script help works!"
    else
        print_error "Script help failed!"
        return 1
    fi
    
    # Test script status (should work even without running containers)
    print_status "Testing script status..."
    if "$script" status > /dev/null 2>&1; then
        print_success "Script status works!"
    else
        print_error "Script status failed!"
        return 1
    fi
    
    return 0
}

# Run all tests
run_all_tests() {
    print_header "MRIA Docker Test Suite"
    
    local failed_tests=0
    local total_tests=0
    
    # Test Docker script
    ((total_tests++))
    if ! test_docker_script; then
        ((failed_tests++))
        print_error "Docker script test failed!"
    fi
    
    # Test standard Docker setup
    ((total_tests++))
    if ! test_standard_docker; then
        ((failed_tests++))
        print_error "Standard Docker test failed!"
    fi
    
    # Test UV Docker setup
    ((total_tests++))
    if ! test_uv_docker; then
        ((failed_tests++))
        print_error "UV Docker test failed!"
    fi
    
    # Test UV performance
    ((total_tests++))
    if ! test_uv_performance; then
        ((failed_tests++))
        print_error "UV performance test failed!"
    fi
    
    # Test production Docker setup (optional, may take longer)
    if [[ "${TEST_PRODUCTION:-false}" == "true" ]]; then
        ((total_tests++))
        if ! test_production_docker; then
            ((failed_tests++))
            print_error "Production Docker test failed!"
        fi
    else
        print_warning "Skipping production test (set TEST_PRODUCTION=true to enable)"
    fi
    
    # Summary
    print_header "Test Results"
    local passed_tests=$((total_tests - failed_tests))
    
    if [[ $failed_tests -eq 0 ]]; then
        print_success "All $total_tests tests passed! 🎉"
        echo -e "\n${GREEN}Docker setup is working correctly!${NC}"
        echo -e "You can now use:"
        echo -e "  ${BLUE}./scripts/docker-uv.sh dev start${NC}  - For development"
        echo -e "  ${BLUE}./scripts/docker-uv.sh prod start${NC} - For production"
    else
        print_error "$failed_tests out of $total_tests tests failed!"
        echo -e "\n${RED}Some Docker configurations need attention.${NC}"
        echo -e "Check the logs above for details."
        return 1
    fi
    
    return 0
}

# Individual test functions
test_individual() {
    case "$1" in
        "standard") test_standard_docker ;;
        "uv") test_uv_docker ;;
        "production") test_production_docker ;;
        "performance") test_uv_performance ;;
        "script") test_docker_script ;;
        *) 
            print_error "Unknown test: $1"
            echo "Available tests: standard, uv, production, performance, script"
            return 1
            ;;
    esac
}

# Help function
show_help() {
    echo "MRIA Docker Test Suite"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  all                - Run all tests (default)"
    echo "  standard           - Test standard Docker setup"
    echo "  uv                 - Test UV Docker setup"
    echo "  production         - Test production Docker setup"
    echo "  performance        - Test UV performance"
    echo "  script             - Test Docker management script"
    echo "  help               - Show this help"
    echo ""
    echo "Environment variables:"
    echo "  TEST_PRODUCTION=true   - Enable production tests in 'all' mode"
    echo "  TEST_TIMEOUT=120       - Test timeout in seconds"
    echo ""
    echo "Examples:"
    echo "  $0                     # Run all tests"
    echo "  $0 uv                  # Test only UV setup"
    echo "  TEST_PRODUCTION=true $0 all  # Run all tests including production"
}

# Main function
main() {
    case "${1:-all}" in
        "all") run_all_tests ;;
        "help"|"--help"|"-h") show_help ;;
        *) test_individual "$1" ;;
    esac
}

# Ensure we're in the right directory
if [[ ! -f "requirements.txt" ]]; then
    print_error "This script must be run from the project root directory!"
    exit 1
fi

# Run main function
main "$@"
