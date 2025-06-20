#!/usr/bin/env python3
"""
Simple system monitoring for local MRIA development.
Monitors Redis, application health, and queue status.
"""

import redis
import requests
import time
import sys
from datetime import datetime

def check_redis():
    """Check Redis connection and basic stats."""
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        info = r.info()
        return {
            'status': 'healthy',
            'connected_clients': info.get('connected_clients', 0),
            'used_memory': info.get('used_memory_human', '0B'),
            'uptime': info.get('uptime_in_seconds', 0)
        }
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}

def check_application():
    """Check application health."""
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            return {'status': 'healthy', 'data': response.json()}
        else:
            return {'status': 'unhealthy', 'code': response.status_code}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}

def check_queue():
    """Check supervisor queue status."""
    try:
        response = requests.get('http://localhost:8000/supervisor/queue/status', timeout=5)
        if response.status_code == 200:
            return {'status': 'healthy', 'data': response.json()}
        else:
            return {'status': 'unhealthy', 'code': response.status_code}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}

def print_status(title, status_data):
    """Print formatted status information."""
    status = status_data.get('status', 'unknown')
    icon = '✅' if status == 'healthy' else '❌'
    print(f"{icon} {title}: {status}")
    
    if 'error' in status_data:
        print(f"   Error: {status_data['error']}")
    elif 'data' in status_data:
        data = status_data['data']
        if title == 'Application':
            components = data.get('components', {})
            for component, comp_status in components.items():
                comp_icon = '✅' if comp_status == 'healthy' else '❌'
                print(f"   {comp_icon} {component}: {comp_status}")
        elif title == 'Queue':
            print(f"   Total Jobs: {data.get('total_jobs', 0)}")
            print(f"   Queued: {data.get('queued_jobs', 0)}")
            print(f"   Running: {data.get('running_jobs', 0)}")
            print(f"   Completed: {data.get('completed_jobs', 0)}")
            print(f"   Failed: {data.get('failed_jobs', 0)}")
    elif title == 'Redis':
        print(f"   Connected Clients: {status_data.get('connected_clients', 0)}")
        print(f"   Memory Used: {status_data.get('used_memory', '0B')}")
        print(f"   Uptime: {status_data.get('uptime', 0)}s")

def monitor_once():
    """Run monitoring once and print results."""
    print(f"\n=== MRIA System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    redis_status = check_redis()
    app_status = check_application()
    queue_status = check_queue()
    
    print_status('Redis', redis_status)
    print_status('Application', app_status)
    print_status('Queue', queue_status)
    
    # Overall health
    all_healthy = all(s.get('status') == 'healthy' for s in [redis_status, app_status, queue_status])
    overall_icon = '🟢' if all_healthy else '🔴'
    print(f"\n{overall_icon} Overall System: {'Healthy' if all_healthy else 'Issues Detected'}")
    
    return all_healthy

def monitor_continuous():
    """Run continuous monitoring."""
    print("🔍 Starting continuous monitoring (Press Ctrl+C to stop)...")
    
    try:
        while True:
            monitor_once()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

def main():
    """Main monitoring function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        monitor_continuous()
    else:
        healthy = monitor_once()
        sys.exit(0 if healthy else 1)

if __name__ == "__main__":
    main()
