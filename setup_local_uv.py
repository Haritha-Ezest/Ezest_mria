#!/usr/bin/env python3
"""
UV-optimized setup script for local MRIA development.
Uses UV for fast Python package management.
"""

import os
import sys
import subprocess
import redis
import time
import shutil
from pathlib import Path

def check_command(command):
    """Check if a command is available."""
    try:
        subprocess.run([command, '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_uv():
    """Check UV installation."""
    print("⚡ Checking UV...")
    
    if not check_command('uv'):
        print("❌ UV not found. Installing UV...")
        try:
            # Try to install UV
            if os.name == 'nt':  # Windows
                subprocess.run(['powershell', '-c', 'irm https://astral.sh/uv/install.ps1 | iex'], check=True)
            else:  # Unix-like
                subprocess.run(['curl', '-LsSf', 'https://astral.sh/uv/install.sh'], stdout=subprocess.PIPE, check=True)
                subprocess.run(['sh'], input=subprocess.run(['curl', '-LsSf', 'https://astral.sh/uv/install.sh'], 
                                                          capture_output=True, check=True).stdout, check=True)
            
            # Check again after installation
            if check_command('uv'):
                print("✅ UV installed successfully")
                return True
            else:
                print("❌ UV installation failed. Please install manually:")
                print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
                return False
        except Exception as e:
            print(f"❌ Failed to install UV: {e}")
            print("   Please install manually: curl -LsSf https://astral.sh/uv/install.sh | sh")
            return False
    else:
        print("✅ UV found")
        return True

def check_redis():
    """Check Redis installation and connection."""
    print("🔍 Checking Redis...")
    
    # Check if redis-cli is available
    if not check_command('redis-cli'):
        print("❌ Redis CLI not found. Please install Redis:")
        print("   Ubuntu/Debian: sudo apt install redis-server")
        print("   macOS: brew install redis")
        print("   Windows: choco install redis-64")
        return False
    
    # Try to connect to Redis
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis is running and accessible")
        return True
    except redis.ConnectionError:
        print("❌ Redis is not running. Starting Redis...")
        
        # Try to start Redis
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['redis-server'], check=True)
            else:  # Unix-like
                subprocess.run(['redis-server', '--daemonize', 'yes'], check=True)
            
            # Wait and test again
            time.sleep(2)
            r.ping()
            print("✅ Redis started successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to start Redis: {e}")
            print("   Please start Redis manually:")
            print("   - Linux: sudo systemctl start redis-server")
            print("   - macOS: brew services start redis")
            print("   - Windows: Start Redis service")
            return False

def setup_python_env_with_uv():
    """Set up Python environment using UV."""
    print("⚡ Setting up Python environment with UV...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required, found {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} found")
    
    # Remove existing venv if it exists and is corrupted
    venv_path = Path('.venv')
    if venv_path.exists():
        # Check if it's a valid UV venv
        uv_marker = venv_path / 'pyvenv.cfg'
        if not uv_marker.exists():
            print("🔧 Removing corrupted virtual environment...")
            shutil.rmtree(venv_path)
    
    # Create virtual environment with UV
    if not venv_path.exists():
        print("📦 Creating virtual environment with UV...")
        try:
            subprocess.run(['uv', 'venv'], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Install dependencies with UV
    print("⚡ Installing dependencies with UV (this is much faster than pip!)...")
    try:
        # Use UV to install from requirements.txt
        subprocess.run(['uv', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies with UV: {e}")
        # Fallback to regular pip if UV fails
        try:
            print("🔄 Falling back to pip...")
            if os.name == 'nt':
                pip_path = '.venv\\Scripts\\pip'
            else:
                pip_path = '.venv/bin/pip'
            
            subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
            print("✅ Dependencies installed with pip")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ Failed with pip too: {e2}")
            return False

def check_ocr_dependencies():
    """Check OCR-related dependencies."""
    print("🔍 Checking OCR dependencies...")
    
    # Check Tesseract
    if not check_command('tesseract'):
        print("❌ Tesseract not found. Please install:")
        print("   Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-eng")
        print("   macOS: brew install tesseract")
        print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    print("✅ Tesseract OCR found")
    return True

def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    
    directories = [
        'storage/uploads',
        'storage/processed',
        'storage/temp',
        'storage/backup',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return True

def setup_environment():
    """Set up environment file."""
    print("⚙️ Setting up environment...")
    
    if not Path('.env').exists():
        if Path('.env.local').exists():
            shutil.copy('.env.local', '.env')
            print("✅ Environment file created from .env.local")
        else:
            # Create basic .env file
            env_content = """# Local Development Environment
REDIS_URL=redis://localhost:6379
LOG_LEVEL=DEBUG
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-secret-key-for-development
API_HOST=0.0.0.0
API_PORT=8000
"""
            with open('.env', 'w') as f:
                f.write(env_content)
            print("✅ Basic environment file created")
    else:
        print("✅ Environment file already exists")
    
    return True

def test_application():
    """Test if the application can start."""
    print("🧪 Testing application startup...")
    
    # Try to import the main application
    try:
        sys.path.insert(0, '.')
        __import__('app.main')
        print("✅ Application imports successfully")
        return True
    except Exception as e:
        print(f"❌ Application import failed: {e}")
        print("   This might be due to missing dependencies or configuration issues")
        return False

def create_uv_scripts():
    """Create UV-optimized scripts."""
    print("📝 Creating UV-optimized scripts...")
    
    # Update start-dev script to use UV
    start_dev_content = '''#!/bin/bash
set -e

echo "🚀 Starting MRIA development environment with UV..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "📦 Starting Redis..."
    if command -v systemctl > /dev/null 2>&1; then
        sudo systemctl start redis-server
    elif command -v brew > /dev/null 2>&1; then
        brew services start redis
    else
        redis-server --daemonize yes
    fi
    
    # Wait for Redis to start
    sleep 2
    
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis started successfully"
    else
        echo "❌ Failed to start Redis"
        exit 1
    fi
else
    echo "✅ Redis is already running"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Update dependencies with UV if available
if command -v uv > /dev/null 2>&1; then
    echo "⚡ Updating dependencies with UV..."
    uv pip install -r requirements.txt
else
    echo "📦 Updating dependencies with pip..."
    pip install -r requirements.txt
fi

# Create storage directories
echo "📁 Ensuring storage directories exist..."
mkdir -p storage/{uploads,processed,temp,backup}
mkdir -p logs

# Copy local environment if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    if [ -f ".env.local" ]; then
        cp .env.local .env
    else
        cat > .env << EOF
REDIS_URL=redis://localhost:6379
LOG_LEVEL=DEBUG
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-secret-key-for-development
API_HOST=0.0.0.0
API_PORT=8000
EOF
    fi
fi

# Start the application with hot reload
echo "🎯 Starting MRIA application..."
echo "📍 Application will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Interactive API: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the application"

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env
'''
    
    with open('scripts/start-dev-uv.sh', 'w') as f:
        f.write(start_dev_content)
    
    # Make executable
    os.chmod('scripts/start-dev-uv.sh', 0o755)
    
    print("✅ UV-optimized scripts created")
    return True

def main():
    """Main setup function."""
    print("⚡ MRIA Local Development Setup with UV")
    print("=" * 60)
    
    checks = [
        ("UV Package Manager", check_uv),
        ("Redis", check_redis),
        ("Python Environment (UV)", setup_python_env_with_uv),
        ("OCR Dependencies", check_ocr_dependencies),
        ("Directories", setup_directories),
        ("Environment", setup_environment),
        ("UV Scripts", create_uv_scripts),
        ("Application", test_application)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📋 {name}")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 Setup completed successfully with UV!")
        print("\nNext steps:")
        print("1. Start the development server:")
        print("   ./scripts/start-dev-uv.sh  (UV-optimized)")
        print("   OR")
        print("   ./scripts/start-docker.sh  (Docker)")
        print("\n2. Open the application:")
        print("   http://localhost:8000")
        print("   http://localhost:8000/docs (API documentation)")
        print("\n3. Run tests:")
        print("   ./scripts/test-system.sh")
        print("\n4. Monitor system:")
        print("   python scripts/monitor.py")
        print("\n⚡ UV Benefits:")
        print("   - 10-100x faster dependency installation")
        print("   - Better dependency resolution")
        print("   - Improved caching")
    else:
        print("❌ Setup failed. Please fix the issues above and run again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
