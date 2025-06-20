#!/bin/bash
set -e

echo "🧪 Running MRIA system tests..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if application is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Application is not running. Please start it first with: scripts/start-dev.sh"
    exit 1
fi

echo "✅ Application is running"

# Run validation tests
echo "🔍 Running system validation..."
python validate_system.py

# Run basic API tests
echo "🌐 Testing API endpoints..."

echo "  - Testing health endpoint..."
curl -s http://localhost:8000/health | jq .

echo "  - Testing OCR health endpoint..."
curl -s http://localhost:8000/ocr/health | jq .

echo "  - Testing supervisor queue status..."
curl -s http://localhost:8000/supervisor/queue/status | jq .

echo ""
echo "✅ Basic API tests completed"

# Optionally run comprehensive integration tests
read -p "🔬 Do you want to run comprehensive integration tests? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔬 Running comprehensive integration tests..."
    python comprehensive_integration_test.py
fi

echo "🎉 System tests completed successfully!"
