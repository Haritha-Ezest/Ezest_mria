#!/usr/bin/env python3
"""
Quick setup and test script for local MRIA development.
"""

import os
import sys
import subprocess
import redis
import time
from pathlib import Path

def check_command(command):
    """Check if a command is available."""
    try:
        subprocess.run([command, '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

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

def check_python_env():
    """Check Python environment and dependencies."""
    print("🐍 Checking Python environment...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required, found {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} found")
    
    # Check if virtual environment exists and has pip
    venv_path = Path('.venv')
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
    
    # Determine paths based on OS
    if os.name == 'nt':
        pip_path = '.venv\\Scripts\\pip'
        pip_exe = Path(pip_path)
    else:
        pip_path = '.venv/bin/pip'
        pip_exe = Path(pip_path)
    
    # Check if pip exists in venv, if not recreate venv
    if not pip_exe.exists():
        print("🔧 Virtual environment incomplete, recreating...")
        import shutil
        shutil.rmtree('.venv', ignore_errors=True)
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
    
    # Install requirements
    print("📦 Installing/updating dependencies...")
    try:
        subprocess.run([str(pip_exe), 'install', '-r', 'requirements.txt'], check=True, capture_output=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        # Try with system python as fallback
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True, capture_output=True)
            print("✅ Dependencies installed (using system pip)")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ Failed with system pip too: {e2}")
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
            import shutil
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
        __import__('app.main')  # Import without assignment
        print("✅ Application imports successfully")
        return True
    except Exception as e:
        print(f"❌ Application import failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 MRIA Local Development Setup")
    print("=" * 50)
    
    checks = [
        ("Python Environment", check_python_env),
        ("Redis", check_redis),
        ("OCR Dependencies", check_ocr_dependencies),
        ("Directories", setup_directories),
        ("Environment", setup_environment),
        ("Application", test_application)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📋 {name}")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the development server:")
        print("   ./scripts/start-dev.sh")
        print("   OR")
        print("   ./scripts/start-docker.sh")
        print("\n2. Open the application:")
        print("   http://localhost:8000")
        print("   http://localhost:8000/docs (API documentation)")
        print("\n3. Run tests:")
        print("   ./scripts/test-system.sh")
        print("\n4. Monitor system:")
        print("   python scripts/monitor.py")
    else:
        print("❌ Setup failed. Please fix the issues above and run again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
