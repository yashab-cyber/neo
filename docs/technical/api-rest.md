# REST API Reference

## Overview

NEO provides a comprehensive REST API that allows external applications to interact with all core functionality. This API follows RESTful principles and provides JSON responses for all endpoints.

## Base URL and Versioning

```
Base URL: https://your-neo-instance.com/api/v1
Content-Type: application/json
Accept: application/json
```

## Authentication

### API Key Authentication

```bash
# Include API key in header
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://your-neo-instance.com/api/v1/status
```

### OAuth 2.0 Authentication

```bash
# Get access token
curl -X POST https://your-neo-instance.com/oauth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "client_credentials",
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "scope": "neo:execute neo:read"
  }'

# Use access token
curl -H "Authorization: Bearer ACCESS_TOKEN" \
     https://your-neo-instance.com/api/v1/commands
```

## Core Endpoints

### System Information

#### GET /api/v1/status
Get system status and health information.

```json
{
  "status": "healthy",
  "version": "2.1.0",
  "uptime": "72h 15m 30s",
  "system": {
    "cpu_usage": 23.5,
    "memory_usage": 45.2,
    "disk_usage": 67.8
  },
  "services": {
    "ai_engine": "running",
    "database": "running",
    "scheduler": "running"
  }
}
```

#### GET /api/v1/system/info
Get detailed system information.

```json
{
  "hostname": "neo-server-01",
  "platform": "linux",
  "architecture": "x86_64",
  "cpu_cores": 8,
  "total_memory": "32GB",
  "neo_version": "2.1.0",
  "python_version": "3.11.0",
  "ai_models": [
    {
      "name": "gpt-4-turbo",
      "status": "loaded",
      "memory_usage": "2.4GB"
    }
  ]
}
```

### Command Execution

#### POST /api/v1/commands/execute
Execute a NEO command.

**Request:**
```json
{
  "command": "system.get_disk_usage",
  "parameters": {
    "path": "/home/user",
    "format": "human_readable"
  },
  "async": false,
  "timeout": 30
}
```

**Response:**
```json
{
  "execution_id": "exec_abc123",
  "status": "completed",
  "result": {
    "total": "500GB",
    "used": "350GB",
    "available": "150GB",
    "percentage": 70
  },
  "execution_time": 1.23,
  "timestamp": "2025-06-30T10:30:00Z"
}
```

#### POST /api/v1/commands/batch
Execute multiple commands in batch.

**Request:**
```json
{
  "commands": [
    {
      "id": "cmd1",
      "command": "system.cpu_usage"
    },
    {
      "id": "cmd2",
      "command": "system.memory_usage"
    }
  ],
  "parallel": true,
  "fail_fast": false
}
```

#### GET /api/v1/commands/{execution_id}
Get execution status and results.

```json
{
  "execution_id": "exec_abc123",
  "status": "running",
  "progress": 65,
  "started_at": "2025-06-30T10:30:00Z",
  "estimated_completion": "2025-06-30T10:35:00Z"
}
```

### AI Services

#### POST /api/v1/ai/chat
Chat with NEO's AI engine.

**Request:**
```json
{
  "message": "Analyze the system performance and suggest optimizations",
  "context": {
    "include_system_metrics": true,
    "include_recent_logs": true
  },
  "model": "gpt-4-turbo",
  "stream": false
}
```

**Response:**
```json
{
  "response": "Based on current system metrics, I recommend...",
  "confidence": 0.92,
  "tokens_used": 150,
  "processing_time": 2.1,
  "suggestions": [
    {
      "action": "optimize_memory",
      "confidence": 0.85,
      "command": "system.optimize_memory"
    }
  ]
}
```

#### POST /api/v1/ai/analyze
Analyze data or files with AI.

**Request:**
```json
{
  "type": "log_analysis",
  "data": {
    "log_files": ["/var/log/neo.log"],
    "time_range": "last_24h"
  },
  "analysis_type": "anomaly_detection"
}
```

### File Management

#### GET /api/v1/files
List files and directories.

**Query Parameters:**
- `path` (string): Directory path
- `recursive` (boolean): Include subdirectories
- `filter` (string): File pattern filter
- `limit` (integer): Maximum results
- `offset` (integer): Pagination offset

**Response:**
```json
{
  "path": "/home/user/documents",
  "total_items": 25,
  "items": [
    {
      "name": "report.pdf",
      "type": "file",
      "size": 2048576,
      "modified": "2025-06-30T09:15:00Z",
      "permissions": "rw-r--r--"
    }
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "has_more": true
  }
}
```

#### POST /api/v1/files/upload
Upload files to NEO.

**Request (multipart/form-data):**
```
file: [binary data]
path: /uploads/
overwrite: false
```

#### GET /api/v1/files/download
Download files from NEO.

**Query Parameters:**
- `path` (required): File path
- `inline` (boolean): Display inline vs download

### Task Management

#### GET /api/v1/tasks
List scheduled and running tasks.

```json
{
  "tasks": [
    {
      "id": "task_123",
      "name": "Daily Backup",
      "status": "scheduled",
      "next_run": "2025-07-01T02:00:00Z",
      "last_run": "2025-06-30T02:00:00Z",
      "success_rate": 98.5
    }
  ],
  "total": 15,
  "running": 3,
  "scheduled": 12
}
```

#### POST /api/v1/tasks
Create a new task.

**Request:**
```json
{
  "name": "Custom Backup Task",
  "command": "backup.create",
  "parameters": {
    "source": "/important/data",
    "destination": "/backups/"
  },
  "schedule": "0 2 * * *",
  "enabled": true,
  "notifications": {
    "on_success": false,
    "on_failure": true,
    "email": "admin@company.com"
  }
}
```

#### PUT /api/v1/tasks/{task_id}
Update an existing task.

#### DELETE /api/v1/tasks/{task_id}
Delete a task.

#### POST /api/v1/tasks/{task_id}/run
Manually trigger task execution.

### Security

#### GET /api/v1/security/audit
Get security audit information.

```json
{
  "audit_events": [
    {
      "timestamp": "2025-06-30T10:30:00Z",
      "event_type": "authentication",
      "user": "john_doe",
      "source_ip": "192.168.1.100",
      "status": "success"
    }
  ],
  "security_score": 85,
  "recommendations": [
    "Enable two-factor authentication for all users",
    "Update system passwords"
  ]
}
```

#### POST /api/v1/security/scan
Initiate security scan.

**Request:**
```json
{
  "scan_type": "vulnerability",
  "targets": ["system", "network", "applications"],
  "async": true
}
```

### Monitoring and Metrics

#### GET /api/v1/metrics
Get system metrics.

**Query Parameters:**
- `metric` (string): Specific metric name
- `start_time` (string): Start time (ISO 8601)
- `end_time` (string): End time (ISO 8601)
- `granularity` (string): Data granularity (minute, hour, day)

```json
{
  "metrics": {
    "cpu_usage": {
      "current": 23.5,
      "average": 18.2,
      "peak": 67.8,
      "data_points": [
        {
          "timestamp": "2025-06-30T10:30:00Z",
          "value": 23.5
        }
      ]
    }
  },
  "time_range": {
    "start": "2025-06-30T09:30:00Z",
    "end": "2025-06-30T10:30:00Z"
  }
}
```

## WebSocket API

### Real-time Updates

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('wss://your-neo-instance.com/api/v1/ws');

ws.onopen = function() {
  // Subscribe to events
  ws.send(JSON.stringify({
    action: 'subscribe',
    events: ['system.metrics', 'task.status', 'security.alerts']
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Event Types

- `system.metrics`: Real-time system metrics
- `task.status`: Task execution updates
- `security.alerts`: Security event notifications
- `command.progress`: Command execution progress
- `ai.response`: AI processing updates

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "The 'path' parameter is required",
    "details": {
      "parameter": "path",
      "expected": "string",
      "received": "null"
    },
    "request_id": "req_abc123"
  }
}
```

### Common Error Codes

- `AUTHENTICATION_FAILED`: Invalid or missing authentication
- `AUTHORIZATION_DENIED`: Insufficient permissions
- `INVALID_PARAMETER`: Invalid request parameter
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server-side error
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

## Rate Limiting

API endpoints are rate-limited based on your plan:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1625097600
```

## SDK Examples

### Python SDK

```python
from neo_sdk import NEOClient

# Initialize client
client = NEOClient(
    base_url="https://your-neo-instance.com",
    api_key="your_api_key"
)

# Execute command
result = client.execute_command(
    "system.get_disk_usage",
    parameters={"path": "/home/user"}
)

# Chat with AI
response = client.ai.chat(
    "What's the current system status?",
    include_context=True
)

# Upload file
client.files.upload(
    file_path="/local/file.txt",
    destination="/remote/file.txt"
)
```

### JavaScript SDK

```javascript
import { NEOClient } from 'neo-js-sdk';

const client = new NEOClient({
  baseUrl: 'https://your-neo-instance.com',
  apiKey: 'your_api_key'
});

// Execute command
const result = await client.executeCommand('system.cpu_usage');

// Real-time metrics
client.subscribe('system.metrics', (data) => {
  console.log('Metrics update:', data);
});
```

This REST API provides comprehensive access to all NEO functionality, enabling seamless integration with external applications and services.
