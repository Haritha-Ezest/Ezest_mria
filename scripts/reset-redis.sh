#!/bin/bash
set -e

echo "🧹 Resetting Redis data..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first."
    exit 1
fi

# Flush all Redis data
redis-cli FLUSHALL

echo "✅ Redis data cleared successfully"

# Optionally restart the application
read -p "🔄 Do you want to restart the MRIA application? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Please restart the application manually with: scripts/start-dev.sh"
fi
