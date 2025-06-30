# Appendix B: API Documentation
**Complete RESTful API Reference for NEO Integration**

---

## B.1 API Overview

NEO provides a comprehensive RESTful API that enables integration with external applications, services, and custom developments. The API follows OpenAPI 3.0 specifications and provides consistent, secure access to all NEO capabilities.

### Base URL and Versioning
```
Base URL: https://api.neo-ai.com/v1
API Version: 1.0
Authentication: Bearer Token, API Key, OAuth 2.0
Rate Limiting: 1000 requests/hour (standard), 10000 requests/hour (enterprise)
```

### API Design Principles
- **RESTful**: Follows REST architectural constraints
- **Stateless**: Each request contains all necessary information
- **Cacheable**: Responses indicate cacheability
- **Uniform Interface**: Consistent naming and structure
- **Secure**: All endpoints require authentication
- **Versioned**: Backward compatibility maintained

## B.2 Authentication

### API Key Authentication
```http
GET /api/v1/system/status
Authorization: Bearer your-api-key-here
Content-Type: application/json
```

### OAuth 2.0 Flow
```http
# Step 1: Authorization
GET /oauth/authorize?client_id=your_client_id&response_type=code&scope=read write

# Step 2: Token Exchange
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=authorization_code_here&
client_id=your_client_id&
client_secret=your_client_secret
```

### JWT Token Structure
```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user123",
    "iss": "neo-ai.com",
    "aud": "api.neo-ai.com",
    "exp": 1672531199,
    "iat": 1672444799,
    "scope": ["read", "write", "admin"]
  }
}
```

## B.3 Core System APIs

### System Status
```http
GET /api/v1/system/status
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 86400,
  "services": {
    "ai_engine": "operational",
    "security": "operational",
    "learning": "operational"
  },
  "performance": {
    "cpu_usage": 15.2,
    "memory_usage": 34.7,
    "response_time_ms": 245
  }
}
```

### System Information
```http
GET /api/v1/system/info
```

**Response:**
```json
{
  "system": {
    "os": "Linux",
    "version": "Ubuntu 20.04.3 LTS",
    "architecture": "x86_64",
    "hostname": "neo-server-01"
  },
  "hardware": {
    "cpu": {
      "model": "Intel Core i7-10700K",
      "cores": 8,
      "threads": 16,
      "frequency_ghz": 3.8
    },
    "memory": {
      "total_gb": 32,
      "available_gb": 20.8,
      "used_percent": 35.0
    },
    "storage": {
      "total_gb": 1024,
      "used_gb": 456,
      "available_gb": 568
    }
  }
}
```

## B.4 AI and Learning APIs

### Query Processing
```http
POST /api/v1/ai/query
Content-Type: application/json

{
  "query": "What is the capital of France?",
  "context": {
    "user_id": "user123",
    "session_id": "session456"
  },
  "options": {
    "include_sources": true,
    "max_response_length": 500,
    "language": "en"
  }
}
```

**Response:**
```json
{
  "response": "The capital of France is Paris. It is located in the north-central part of the country...",
  "confidence": 0.95,
  "sources": [
    {
      "type": "knowledge_base",
      "reliability": 0.98
    }
  ],
  "processing_time_ms": 234,
  "session_id": "session456"
}
```

### Learning Management
```http
POST /api/v1/ai/learn
Content-Type: application/json

{
  "data": {
    "interaction": "User asked about Python best practices",
    "response": "Provided PEP 8 guidelines and examples",
    "feedback": "positive",
    "user_context": {
      "skill_level": "intermediate",
      "domain": "software_development"
    }
  }
}
```

### Model Training
```http
POST /api/v1/ai/models/train
Content-Type: application/json

{
  "model_name": "custom_classifier",
  "training_data": {
    "dataset_url": "https://storage.neo-ai.com/datasets/custom_data.json",
    "format": "json",
    "labels": ["category_a", "category_b", "category_c"]
  },
  "parameters": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

## B.5 Security APIs

### Security Scan
```http
POST /api/v1/security/scan
Content-Type: application/json

{
  "target": {
    "type": "system",
    "scope": "full"
  },
  "options": {
    "scan_type": "vulnerability",
    "depth": "standard",
    "include_compliance": true
  }
}
```

**Response:**
```json
{
  "scan_id": "scan_789",
  "status": "completed",
  "start_time": "2025-06-29T10:00:00Z",
  "end_time": "2025-06-29T10:15:00Z",
  "results": {
    "vulnerabilities_found": 3,
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 0
  },
  "compliance": {
    "pci_dss": "compliant",
    "gdpr": "compliant",
    "iso27001": "partial"
  }
}
```

### Threat Detection
```http
GET /api/v1/security/threats/active
```

**Response:**
```json
{
  "active_threats": [
    {
      "threat_id": "threat_001",
      "type": "suspicious_network_activity",
      "severity": "medium",
      "description": "Unusual outbound traffic detected",
      "first_detected": "2025-06-29T09:45:00Z",
      "source_ip": "192.168.1.100",
      "destination": "suspicious-domain.com",
      "status": "monitoring"
    }
  ],
  "total_threats": 1
}
```

### Incident Management
```http
POST /api/v1/security/incidents
Content-Type: application/json

{
  "title": "Suspicious file detected",
  "description": "Potential malware found in downloads folder",
  "severity": "high",
  "category": "malware",
  "affected_systems": ["workstation-01"],
  "reporter": "automated_scan"
}
```

## B.6 System Control APIs

### Process Management
```http
GET /api/v1/system/processes
```

**Response:**
```json
{
  "processes": [
    {
      "pid": 1234,
      "name": "firefox",
      "cpu_percent": 15.2,
      "memory_mb": 512,
      "status": "running",
      "user": "john_doe"
    }
  ],
  "total_processes": 156
}
```

### File Operations
```http
POST /api/v1/system/files/operations
Content-Type: application/json

{
  "operation": "copy",
  "source": "/home/user/documents/file.txt",
  "destination": "/backup/file.txt",
  "options": {
    "preserve_permissions": true,
    "create_backup": true
  }
}
```

### Power Management
```http
POST /api/v1/system/power
Content-Type: application/json

{
  "action": "shutdown",
  "delay_seconds": 300,
  "message": "System will shutdown in 5 minutes for maintenance"
}
```

## B.7 Development APIs

### Code Analysis
```http
POST /api/v1/dev/analyze
Content-Type: application/json

{
  "code": "def calculate_factorial(n):\n    if n == 0:\n        return 1\n    return n * calculate_factorial(n-1)",
  "language": "python",
  "analysis_type": ["security", "performance", "style"]
}
```

**Response:**
```json
{
  "analysis_id": "analysis_456",
  "results": {
    "security": {
      "issues": [],
      "score": 95
    },
    "performance": {
      "issues": [
        {
          "type": "recursion_depth",
          "line": 4,
          "description": "Potential stack overflow for large values of n",
          "suggestion": "Consider iterative approach for better performance"
        }
      ],
      "score": 75
    },
    "style": {
      "issues": [],
      "score": 100
    }
  }
}
```

### Test Generation
```http
POST /api/v1/dev/generate-tests
Content-Type: application/json

{
  "source_code": "function add(a, b) { return a + b; }",
  "language": "javascript",
  "test_framework": "jest",
  "coverage_target": 100
}
```

## B.8 Research APIs

### Research Query
```http
POST /api/v1/research/query
Content-Type: application/json

{
  "query": "Latest developments in quantum computing",
  "domain": "technology",
  "sources": ["academic_papers", "news", "patents"],
  "date_range": {
    "start": "2024-01-01",
    "end": "2025-06-29"
  }
}
```

### Data Analysis
```http
POST /api/v1/research/analyze
Content-Type: multipart/form-data

dataset: [binary file data]
analysis_type: statistical
parameters: {"confidence_level": 0.95, "test_type": "t_test"}
```

## B.9 Automation APIs

### Workflow Creation
```http
POST /api/v1/automation/workflows
Content-Type: application/json

{
  "name": "daily_backup",
  "description": "Daily backup routine",
  "steps": [
    {
      "type": "file_operation",
      "action": "backup",
      "source": "/home/user/documents",
      "destination": "/backup/daily"
    },
    {
      "type": "notification",
      "message": "Backup completed successfully"
    }
  ],
  "schedule": {
    "frequency": "daily",
    "time": "02:00:00",
    "timezone": "UTC"
  }
}
```

### Task Execution
```http
POST /api/v1/automation/tasks/execute
Content-Type: application/json

{
  "task_id": "task_123",
  "parameters": {
    "target_directory": "/home/user/projects",
    "compression_level": "high"
  }
}
```

## B.10 Monitoring and Metrics APIs

### Performance Metrics
```http
GET /api/v1/monitoring/metrics?start_time=2025-06-29T00:00:00Z&end_time=2025-06-29T23:59:59Z
```

**Response:**
```json
{
  "metrics": {
    "cpu_usage": {
      "average": 25.5,
      "peak": 78.2,
      "data_points": [
        {"timestamp": "2025-06-29T00:00:00Z", "value": 15.2},
        {"timestamp": "2025-06-29T00:05:00Z", "value": 18.7}
      ]
    },
    "memory_usage": {
      "average": 45.3,
      "peak": 67.8
    }
  }
}
```

### Alert Configuration
```http
POST /api/v1/monitoring/alerts
Content-Type: application/json

{
  "name": "high_cpu_usage",
  "condition": {
    "metric": "cpu_usage",
    "operator": "greater_than",
    "threshold": 80,
    "duration": "5m"
  },
  "actions": [
    {
      "type": "email",
      "recipients": ["admin@example.com"]
    },
    {
      "type": "webhook",
      "url": "https://hooks.slack.com/webhook"
    }
  ]
}
```

## B.11 Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request contains invalid parameters",
    "details": {
      "field": "query",
      "issue": "Query cannot be empty"
    },
    "request_id": "req_789",
    "timestamp": "2025-06-29T10:30:00Z"
  }
}
```

### HTTP Status Codes
```
200 OK - Successful request
201 Created - Resource created successfully
400 Bad Request - Invalid request parameters
401 Unauthorized - Authentication required
403 Forbidden - Insufficient permissions
404 Not Found - Resource not found
429 Too Many Requests - Rate limit exceeded
500 Internal Server Error - Server error
503 Service Unavailable - Service temporarily unavailable
```

## B.12 Rate Limiting

### Rate Limit Headers
```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1672531199
X-RateLimit-Window: 3600
```

### Rate Limit Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "API rate limit exceeded",
    "retry_after": 3600
  }
}
```

## B.13 Webhooks

### Webhook Configuration
```http
POST /api/v1/webhooks
Content-Type: application/json

{
  "url": "https://your-server.com/neo-webhook",
  "events": ["security.threat_detected", "system.error", "task.completed"],
  "secret": "your_webhook_secret"
}
```

### Webhook Payload
```json
{
  "event": "security.threat_detected",
  "timestamp": "2025-06-29T10:45:00Z",
  "data": {
    "threat_id": "threat_002",
    "severity": "high",
    "description": "Malware detected in system files"
  },
  "signature": "sha256=abc123def456..."
}
```

---

**Next**: [Appendix C: Configuration Files](appendix-c-config.md)

*This API documentation provides comprehensive access to all NEO capabilities through programmatic interfaces, enabling seamless integration with existing systems and custom applications.*
