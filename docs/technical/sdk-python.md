# Python SDK Documentation

## Overview

The NEO Python SDK provides a comprehensive interface for interacting with NEO from Python applications. It supports both synchronous and asynchronous operations, with full type hints and extensive error handling.

## Installation

```bash
# Install from PyPI
pip install neo-python-sdk

# Install with optional dependencies
pip install neo-python-sdk[async,dev]

# Install from source
git clone https://github.com/neo-ai/python-sdk.git
cd python-sdk
pip install -e .
```

## Quick Start

```python
from neo_sdk import NEOClient
import asyncio

# Synchronous client
client = NEOClient(
    base_url="https://your-neo-instance.com",
    api_key="your_api_key"
)

# Basic command execution
result = client.execute("system.status")
print(result)

# Asynchronous client
async def main():
    async_client = client.async_client()
    result = await async_client.execute("system.status")
    print(result)

asyncio.run(main())
```

## Authentication

### API Key Authentication

```python
from neo_sdk import NEOClient

# Using API key
client = NEOClient(
    base_url="https://your-neo-instance.com",
    api_key="your_api_key"
)

# Using environment variable
import os
os.environ['NEO_API_KEY'] = 'your_api_key'
client = NEOClient(base_url="https://your-neo-instance.com")
```

### OAuth 2.0 Authentication

```python
from neo_sdk import NEOClient, OAuth2Config

# OAuth2 configuration
oauth_config = OAuth2Config(
    client_id="your_client_id",
    client_secret="your_client_secret",
    authorization_url="https://your-neo-instance.com/oauth/authorize",
    token_url="https://your-neo-instance.com/oauth/token",
    scope="neo:execute neo:read"
)

client = NEOClient(
    base_url="https://your-neo-instance.com",
    auth=oauth_config
)

# Handle OAuth flow
auth_url = client.get_authorization_url()
print(f"Visit: {auth_url}")

# After user authorization, exchange code for token
authorization_code = input("Enter authorization code: ")
client.exchange_code_for_token(authorization_code)
```

## Core Client Operations

### Command Execution

```python
from neo_sdk import NEOClient
from neo_sdk.exceptions import NEOError, CommandError

client = NEOClient(base_url="https://your-neo-instance.com", api_key="your_api_key")

# Simple command execution
try:
    result = client.execute("system.cpu_usage")
    print(f"CPU Usage: {result.data}%")
except CommandError as e:
    print(f"Command failed: {e}")

# Command with parameters
result = client.execute(
    "files.search",
    parameters={
        "path": "/home/user",
        "pattern": "*.py",
        "recursive": True
    }
)

# Asynchronous execution
async def async_command():
    result = await client.async_execute("system.analyze_performance")
    return result

# Batch command execution
commands = [
    {"command": "system.cpu_usage"},
    {"command": "system.memory_usage"},
    {"command": "system.disk_usage", "parameters": {"path": "/"}}
]

batch_results = client.execute_batch(commands, parallel=True)
for result in batch_results:
    print(f"Command: {result.command}, Result: {result.data}")
```

### Long-running Commands

```python
# Execute long-running command asynchronously
execution = client.execute_async("backup.create_full", parameters={
    "source": "/important/data",
    "destination": "/backups/full_backup.tar.gz"
})

print(f"Execution ID: {execution.id}")

# Monitor progress
import time
while not execution.is_complete():
    status = execution.get_status()
    print(f"Progress: {status.progress}% - {status.message}")
    time.sleep(5)

# Get final result
result = execution.get_result()
print(f"Backup completed: {result.data}")

# Using context manager for cleanup
with client.execute_async("long_task") as execution:
    # Monitor or do other work
    while not execution.is_complete():
        time.sleep(1)
    result = execution.get_result()
```

## AI Services

### Chat Interface

```python
from neo_sdk.ai import ChatSession

# Create chat session
chat = client.ai.create_chat_session(
    model="gpt-4-turbo",
    system_prompt="You are a helpful system administrator assistant."
)

# Single message
response = chat.send_message(
    "What's the current system status and any recommendations?"
)
print(response.content)

# Conversation with context
chat.send_message("Check the disk usage")
response = chat.send_message("What should I do if it's above 90%?")
print(response.content)

# Streaming responses
for chunk in chat.send_message_stream("Analyze the system logs"):
    print(chunk.content, end='', flush=True)

# Chat with additional context
response = chat.send_message(
    "Optimize system performance",
    context={
        "include_metrics": True,
        "include_logs": True,
        "time_range": "last_1h"
    }
)
```

### AI Analysis

```python
# Analyze files
analysis = client.ai.analyze_file(
    file_path="/var/log/system.log",
    analysis_type="anomaly_detection"
)

print(f"Anomalies found: {len(analysis.anomalies)}")
for anomaly in analysis.anomalies:
    print(f"- {anomaly.description} (confidence: {anomaly.confidence})")

# Analyze system metrics
metrics_analysis = client.ai.analyze_metrics(
    metrics=["cpu_usage", "memory_usage", "disk_io"],
    time_range="24h",
    analysis_type="performance_optimization"
)

for recommendation in metrics_analysis.recommendations:
    print(f"Recommendation: {recommendation.action}")
    print(f"Impact: {recommendation.expected_impact}")
    print(f"Command: {recommendation.command}")

# Custom AI analysis
custom_analysis = client.ai.analyze(
    data={
        "logs": client.files.read("/var/log/app.log"),
        "metrics": client.metrics.get_recent("cpu_usage", hours=24)
    },
    prompt="Identify performance bottlenecks and suggest optimizations",
    model="claude-3"
)
```

## File Management

### File Operations

```python
from neo_sdk.files import FileManager
import io

# File manager instance
files = client.files

# List files
file_list = files.list(
    path="/home/user/documents",
    recursive=True,
    filter_pattern="*.pdf",
    sort_by="modified",
    sort_order="desc"
)

for file_info in file_list:
    print(f"{file_info.name} - {file_info.size} bytes - {file_info.modified}")

# Upload file
with open("local_file.txt", "rb") as f:
    upload_result = files.upload(
        file_data=f,
        destination="/remote/uploaded_file.txt",
        overwrite=True,
        create_dirs=True
    )

# Upload with progress callback
def progress_callback(bytes_uploaded, total_bytes):
    percent = (bytes_uploaded / total_bytes) * 100
    print(f"Upload progress: {percent:.1f}%")

files.upload(
    file_path="large_file.zip",
    destination="/remote/large_file.zip",
    progress_callback=progress_callback
)

# Download file
download_data = files.download("/remote/file.txt")
with open("local_copy.txt", "wb") as f:
    f.write(download_data)

# Stream download for large files
with files.download_stream("/remote/large_file.zip") as stream:
    with open("local_large_file.zip", "wb") as f:
        for chunk in stream:
            f.write(chunk)

# File operations
files.copy("/source/file.txt", "/destination/file.txt")
files.move("/old/location/file.txt", "/new/location/file.txt")
files.delete("/unwanted/file.txt")

# Directory operations
files.create_directory("/new/directory", parents=True)
files.delete_directory("/old/directory", recursive=True)

# File metadata
metadata = files.get_metadata("/important/file.txt")
print(f"Size: {metadata.size}, Modified: {metadata.modified}")
print(f"Permissions: {metadata.permissions}")
print(f"Checksum: {metadata.checksum}")
```

### File Search

```python
# Text-based search
search_results = files.search(
    path="/documents",
    query="project status report",
    search_type="content",
    file_types=["pdf", "docx", "txt"]
)

# Advanced search with filters
advanced_search = files.advanced_search(
    base_path="/",
    filters={
        "size_min": "1MB",
        "size_max": "100MB",
        "modified_after": "2025-01-01",
        "modified_before": "2025-06-30",
        "content_contains": "important data",
        "file_types": ["log", "txt", "json"]
    },
    sort_by="relevance"
)

# Fuzzy search
fuzzy_results = files.fuzzy_search(
    query="sistem confgiuration",  # Intentional typos
    path="/config",
    max_distance=2
)
```

## Task Management

### Task Operations

```python
from neo_sdk.tasks import TaskManager
from datetime import datetime, timedelta

tasks = client.tasks

# List tasks
task_list = tasks.list(
    status="all",  # "running", "scheduled", "completed", "failed"
    limit=50,
    sort_by="created"
)

for task in task_list:
    print(f"Task: {task.name} - Status: {task.status}")

# Create task
new_task = tasks.create(
    name="Daily System Report",
    command="system.generate_report",
    parameters={
        "report_type": "daily",
        "include_metrics": True,
        "email_recipients": ["admin@company.com"]
    },
    schedule="0 9 * * *",  # Daily at 9 AM
    enabled=True,
    retry_policy={
        "max_retries": 3,
        "retry_delay": 300  # 5 minutes
    }
)

print(f"Created task: {new_task.id}")

# Update task
tasks.update(
    task_id=new_task.id,
    name="Updated Daily System Report",
    schedule="0 8 * * *",  # Changed to 8 AM
    parameters={
        "report_type": "comprehensive",
        "include_metrics": True,
        "include_logs": True,
        "email_recipients": ["admin@company.com", "team@company.com"]
    }
)

# Manual task execution
execution = tasks.run(task_id=new_task.id, wait_for_completion=False)
print(f"Task execution started: {execution.id}")

# Monitor execution
while not execution.is_complete():
    status = execution.get_status()
    print(f"Execution progress: {status.progress}%")
    time.sleep(5)

result = execution.get_result()
print(f"Task completed with result: {result}")

# Task scheduling with complex patterns
complex_task = tasks.create(
    name="Weekly Maintenance",
    command="system.maintenance",
    schedule={
        "type": "cron",
        "expression": "0 2 * * 0",  # Sundays at 2 AM
        "timezone": "UTC"
    },
    conditions={
        "system_load": {"max": 0.5},  # Only run if system load < 50%
        "maintenance_window": True
    }
)
```

### Task Dependencies

```python
# Create tasks with dependencies
backup_task = tasks.create(
    name="Database Backup",
    command="database.backup",
    parameters={"database": "main_db"}
)

cleanup_task = tasks.create(
    name="Cleanup Old Backups",
    command="files.cleanup_old",
    parameters={
        "path": "/backups",
        "keep_days": 30
    },
    dependencies=[backup_task.id]  # Run after backup completes
)

report_task = tasks.create(
    name="Backup Report",
    command="reports.generate",
    parameters={
        "report_type": "backup_summary",
        "email": "admin@company.com"
    },
    dependencies=[backup_task.id, cleanup_task.id]  # Run after both complete
)

# Conditional dependencies
conditional_task = tasks.create(
    name="Emergency Notification",
    command="notifications.send_alert",
    dependencies=[{
        "task_id": backup_task.id,
        "condition": "failed"  # Only run if backup fails
    }]
)
```

## System Monitoring

### Metrics Collection

```python
from neo_sdk.monitoring import MetricsCollector
from datetime import datetime, timedelta

metrics = client.metrics

# Get current metrics
current_metrics = metrics.get_current([
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "network_io"
])

for metric_name, value in current_metrics.items():
    print(f"{metric_name}: {value}")

# Historical metrics
end_time = datetime.now()
start_time = end_time - timedelta(hours=24)

historical_data = metrics.get_historical(
    metrics=["cpu_usage", "memory_usage"],
    start_time=start_time,
    end_time=end_time,
    granularity="1h"  # 1-hour intervals
)

# Process historical data
for metric_name, data_points in historical_data.items():
    values = [point.value for point in data_points]
    avg_value = sum(values) / len(values)
    max_value = max(values)
    print(f"{metric_name}: avg={avg_value:.2f}, max={max_value:.2f}")

# Custom metrics
metrics.record_custom_metric(
    name="application_response_time",
    value=150.5,
    tags={"endpoint": "/api/users", "method": "GET"},
    timestamp=datetime.now()
)

# Metric aggregation
aggregated = metrics.aggregate(
    metric="cpu_usage",
    start_time=start_time,
    end_time=end_time,
    aggregation_function="avg",
    group_by="hour"
)
```

### Real-time Monitoring

```python
from neo_sdk.monitoring import RealTimeMonitor

# Real-time metrics monitoring
def metric_callback(metric_name, value, timestamp):
    print(f"{timestamp}: {metric_name} = {value}")

monitor = client.monitoring.create_real_time_monitor(
    metrics=["cpu_usage", "memory_usage"],
    callback=metric_callback,
    interval=5  # Update every 5 seconds
)

# Start monitoring
monitor.start()

# Custom alerting
def alert_callback(alert):
    print(f"ALERT: {alert.message}")
    print(f"Severity: {alert.severity}")
    if alert.severity == "critical":
        # Send emergency notification
        client.notifications.send_emergency(alert.message)

alert_monitor = client.monitoring.create_alert_monitor(
    rules=[
        {
            "metric": "cpu_usage",
            "threshold": 90,
            "condition": "greater_than",
            "duration": "5m"
        },
        {
            "metric": "memory_usage",
            "threshold": 95,
            "condition": "greater_than",
            "duration": "2m"
        }
    ],
    callback=alert_callback
)

alert_monitor.start()

# Stop monitoring
# monitor.stop()
# alert_monitor.stop()
```

## Security Operations

### Security Monitoring

```python
from neo_sdk.security import SecurityManager

security = client.security

# Get security status
security_status = security.get_status()
print(f"Security Score: {security_status.score}/100")
print(f"Active Threats: {len(security_status.active_threats)}")

for threat in security_status.active_threats:
    print(f"- {threat.description} (Severity: {threat.severity})")

# Security audit
audit_results = security.run_audit(
    scope=["authentication", "authorization", "encryption", "network"],
    detailed=True
)

for finding in audit_results.findings:
    print(f"Finding: {finding.title}")
    print(f"Severity: {finding.severity}")
    print(f"Recommendation: {finding.recommendation}")

# Security scan
scan_result = security.scan(
    scan_type="vulnerability",
    targets=["system", "applications", "network"],
    async_scan=True
)

# Monitor scan progress
while not scan_result.is_complete():
    status = scan_result.get_status()
    print(f"Scan progress: {status.progress}%")
    time.sleep(10)

vulnerabilities = scan_result.get_vulnerabilities()
for vuln in vulnerabilities:
    print(f"Vulnerability: {vuln.title}")
    print(f"CVSS Score: {vuln.cvss_score}")
    print(f"Fix Available: {vuln.fix_available}")
```

### Access Control

```python
# User management
users = security.users

# List users
user_list = users.list(
    include_disabled=False,
    sort_by="last_login"
)

# Create user
new_user = users.create(
    username="john_doe",
    email="john@company.com",
    roles=["user", "developer"],
    temporary_password=True
)

# Update user permissions
users.update_permissions(
    username="john_doe",
    permissions=["system.read", "files.read", "files.write"],
    action="grant"
)

# Role management
roles = security.roles

# Create custom role
custom_role = roles.create(
    name="data_analyst",
    description="Data analysis and reporting access",
    permissions=[
        "data.read",
        "reports.create",
        "reports.read",
        "metrics.read"
    ]
)

# Assign role to user
users.assign_role("john_doe", "data_analyst")
```

## Error Handling and Debugging

### Exception Handling

```python
from neo_sdk.exceptions import (
    NEOError,
    AuthenticationError,
    AuthorizationError,
    CommandError,
    NetworkError,
    RateLimitError
)

try:
    result = client.execute("complex_operation", parameters={
        "data": large_dataset,
        "timeout": 3600
    })
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Handle re-authentication
    client.refresh_authentication()
except AuthorizationError as e:
    print(f"Permission denied: {e}")
    # Request additional permissions
except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after: {e.retry_after} seconds")
    time.sleep(e.retry_after)
    # Retry operation
except CommandError as e:
    print(f"Command failed: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Details: {e.details}")
except NetworkError as e:
    print(f"Network error: {e}")
    # Implement retry logic with exponential backoff
except NEOError as e:
    print(f"General NEO error: {e}")
```

### Debugging and Logging

```python
import logging
from neo_sdk import NEOClient

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create client with debug mode
client = NEOClient(
    base_url="https://your-neo-instance.com",
    api_key="your_api_key",
    debug=True,
    timeout=30,
    retry_config={
        "max_retries": 3,
        "backoff_factor": 2,
        "status_forcelist": [500, 502, 503, 504]
    }
)

# Request logging
client.enable_request_logging(
    log_level="DEBUG",
    log_body=True,
    log_headers=True
)

# Performance monitoring
with client.performance_monitor() as monitor:
    result = client.execute("complex_operation")
    
print(f"Operation took: {monitor.duration}s")
print(f"Network time: {monitor.network_time}s")
print(f"Processing time: {monitor.processing_time}s")
```

## Advanced Usage

### Custom Configurations

```python
from neo_sdk import NEOClient, ClientConfig

# Custom configuration
config = ClientConfig(
    base_url="https://your-neo-instance.com",
    api_key="your_api_key",
    timeout=60,
    max_retries=5,
    backoff_factor=2,
    pool_connections=10,
    pool_maxsize=20,
    headers={
        "User-Agent": "MyApp/1.0",
        "X-Custom-Header": "custom-value"
    }
)

client = NEOClient(config=config)

# Custom session configuration
session_config = {
    "verify_ssl": True,
    "trust_env": True,
    "stream": False,
    "cert": "/path/to/client.cert"
}

client.configure_session(session_config)
```

### Plugin Development

```python
from neo_sdk.plugins import BasePlugin

class CustomPlugin(BasePlugin):
    name = "custom_analytics"
    version = "1.0.0"
    
    def __init__(self, client):
        super().__init__(client)
        self.analytics_api = "https://analytics.company.com/api"
    
    def analyze_data(self, data_source, analysis_type="standard"):
        """Custom data analysis method"""
        # Get data from NEO
        if data_source.startswith("neo://"):
            data = self.client.files.read(data_source[6:])
        else:
            data = self.client.execute("data.fetch", {"source": data_source})
        
        # Perform custom analysis
        analysis_result = self._perform_analysis(data, analysis_type)
        
        # Store results back to NEO
        self.client.execute("data.store", {
            "data": analysis_result,
            "location": f"analysis_results/{data_source}_{analysis_type}"
        })
        
        return analysis_result
    
    def _perform_analysis(self, data, analysis_type):
        # Custom analysis logic
        pass

# Register plugin
client.register_plugin(CustomPlugin)

# Use plugin
analysis = client.plugins.custom_analytics.analyze_data(
    "neo://data/sales_data.csv",
    analysis_type="trend_analysis"
)
```

This comprehensive Python SDK documentation provides everything needed to integrate NEO functionality into Python applications, from basic operations to advanced use cases.
