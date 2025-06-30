# Database Schema Documentation

## Overview

NEO uses a hybrid database architecture combining PostgreSQL for relational data, MongoDB for document storage, Redis for caching, and specialized databases for specific use cases. This document details the complete database schema and relationships.

## Database Architecture

### Primary Databases

```yaml
databases:
  postgresql:
    purpose: "Core system data, user management, task scheduling"
    version: "15.0+"
    connection_pool: "20-100 connections"
    
  mongodb:
    purpose: "Logs, metrics, AI model data, large documents"
    version: "6.0+"
    sharding: "enabled for large collections"
    
  redis:
    purpose: "Caching, session storage, real-time data"
    version: "7.0+"
    persistence: "AOF + RDB"
    
  elasticsearch:
    purpose: "Full-text search, log analysis"
    version: "8.0+"
    indices: "time-based rolling indices"
    
  timescaledb:
    purpose: "Time-series metrics and monitoring data"
    version: "2.10+"
    retention: "automated data lifecycle management"
```

## PostgreSQL Schema

### Core System Tables

#### users
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    salt VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    last_login TIMESTAMPTZ,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMPTZ,
    password_changed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    two_factor_enabled BOOLEAN DEFAULT false,
    two_factor_secret VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT username_length CHECK (LENGTH(username) >= 3),
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_last_login ON users(last_login);
CREATE INDEX idx_users_is_active ON users(is_active);
```

#### user_sessions
```sql
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address INET,
    user_agent TEXT,
    device_info JSONB,
    location_info JSONB,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMPTZ NOT NULL,
    last_activity TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT session_token_length CHECK (LENGTH(session_token) >= 32)
);

CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX idx_user_sessions_last_activity ON user_sessions(last_activity);
```

#### roles
```sql
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB NOT NULL DEFAULT '[]',
    is_system_role BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT role_name_format CHECK (name ~* '^[a-z][a-z0-9_]*$')
);

CREATE INDEX idx_roles_name ON roles(name);
CREATE INDEX idx_roles_is_system ON roles(is_system_role);
```

#### user_roles
```sql
CREATE TABLE user_roles (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    assigned_by UUID REFERENCES users(id),
    assigned_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ,
    
    PRIMARY KEY (user_id, role_id)
);

CREATE INDEX idx_user_roles_user_id ON user_roles(user_id);
CREATE INDEX idx_user_roles_role_id ON user_roles(role_id);
CREATE INDEX idx_user_roles_expires_at ON user_roles(expires_at);
```

### Task Management Tables

#### tasks
```sql
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    command VARCHAR(500) NOT NULL,
    parameters JSONB DEFAULT '{}',
    schedule_expression VARCHAR(100),
    schedule_type VARCHAR(20) DEFAULT 'cron',
    timezone VARCHAR(50) DEFAULT 'UTC',
    is_enabled BOOLEAN DEFAULT true,
    is_system_task BOOLEAN DEFAULT false,
    max_retries INTEGER DEFAULT 3,
    retry_delay_seconds INTEGER DEFAULT 300,
    timeout_seconds INTEGER DEFAULT 3600,
    priority INTEGER DEFAULT 5,
    tags JSONB DEFAULT '[]',
    conditions JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_run_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ,
    
    CONSTRAINT valid_priority CHECK (priority BETWEEN 1 AND 10),
    CONSTRAINT valid_schedule_type CHECK (schedule_type IN ('cron', 'interval', 'once', 'manual'))
);

CREATE INDEX idx_tasks_name ON tasks(name);
CREATE INDEX idx_tasks_is_enabled ON tasks(is_enabled);
CREATE INDEX idx_tasks_next_run_at ON tasks(next_run_at);
CREATE INDEX idx_tasks_priority ON tasks(priority);
CREATE INDEX idx_tasks_created_by ON tasks(created_by);
```

#### task_executions
```sql
CREATE TABLE task_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    execution_number INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    exit_code INTEGER,
    output TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    triggered_by VARCHAR(50) DEFAULT 'scheduler',
    executor_node VARCHAR(100),
    context JSONB DEFAULT '{}',
    
    CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout')),
    CONSTRAINT valid_duration CHECK (duration_seconds >= 0)
);

CREATE INDEX idx_task_executions_task_id ON task_executions(task_id);
CREATE INDEX idx_task_executions_status ON task_executions(status);
CREATE INDEX idx_task_executions_started_at ON task_executions(started_at);
CREATE INDEX idx_task_executions_execution_number ON task_executions(task_id, execution_number);
```

#### task_dependencies
```sql
CREATE TABLE task_dependencies (
    parent_task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    dependent_task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    dependency_type VARCHAR(20) DEFAULT 'success',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (parent_task_id, dependent_task_id),
    CONSTRAINT no_self_dependency CHECK (parent_task_id != dependent_task_id),
    CONSTRAINT valid_dependency_type CHECK (dependency_type IN ('success', 'failure', 'completion', 'always'))
);

CREATE INDEX idx_task_dependencies_parent ON task_dependencies(parent_task_id);
CREATE INDEX idx_task_dependencies_dependent ON task_dependencies(dependent_task_id);
```

### Configuration and Settings

#### system_config
```sql
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value JSONB NOT NULL,
    data_type VARCHAR(20) NOT NULL,
    description TEXT,
    is_sensitive BOOLEAN DEFAULT false,
    is_readonly BOOLEAN DEFAULT false,
    validation_rules JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_by UUID REFERENCES users(id),
    
    UNIQUE(category, key),
    CONSTRAINT valid_data_type CHECK (data_type IN ('string', 'integer', 'float', 'boolean', 'json', 'array'))
);

CREATE INDEX idx_system_config_category ON system_config(category);
CREATE INDEX idx_system_config_key ON system_config(key);
CREATE INDEX idx_system_config_is_sensitive ON system_config(is_sensitive);
```

#### api_keys
```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_prefix VARCHAR(10) NOT NULL,
    permissions JSONB DEFAULT '[]',
    rate_limit_per_minute INTEGER DEFAULT 100,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    usage_count BIGINT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT key_prefix_format CHECK (key_prefix ~* '^neo_[a-z0-9]{4,6}$')
);

CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_is_active ON api_keys(is_active);
CREATE INDEX idx_api_keys_expires_at ON api_keys(expires_at);
```

### File Management

#### file_metadata
```sql
CREATE TABLE file_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    path VARCHAR(1000) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(100),
    mime_type VARCHAR(100),
    size_bytes BIGINT NOT NULL,
    checksum_md5 VARCHAR(32),
    checksum_sha256 VARCHAR(64),
    permissions VARCHAR(10),
    owner_user_id UUID REFERENCES users(id),
    is_encrypted BOOLEAN DEFAULT false,
    encryption_algorithm VARCHAR(50),
    is_compressed BOOLEAN DEFAULT false,
    compression_algorithm VARCHAR(50),
    storage_backend VARCHAR(50) DEFAULT 'local',
    storage_path VARCHAR(1000),
    access_count BIGINT DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_size CHECK (size_bytes >= 0),
    CONSTRAINT valid_storage_backend CHECK (storage_backend IN ('local', 's3', 'gcs', 'azure'))
);

CREATE INDEX idx_file_metadata_path ON file_metadata(path);
CREATE INDEX idx_file_metadata_filename ON file_metadata(filename);
CREATE INDEX idx_file_metadata_file_type ON file_metadata(file_type);
CREATE INDEX idx_file_metadata_owner ON file_metadata(owner_user_id);
CREATE INDEX idx_file_metadata_size ON file_metadata(size_bytes);
CREATE INDEX idx_file_metadata_modified_at ON file_metadata(modified_at);
```

#### file_permissions
```sql
CREATE TABLE file_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES file_metadata(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    permission_type VARCHAR(20) NOT NULL,
    granted_by UUID REFERENCES users(id),
    granted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ,
    
    CONSTRAINT valid_permission_type CHECK (permission_type IN ('read', 'write', 'execute', 'delete', 'admin')),
    CONSTRAINT user_or_role_specified CHECK ((user_id IS NOT NULL) != (role_id IS NOT NULL))
);

CREATE INDEX idx_file_permissions_file_id ON file_permissions(file_id);
CREATE INDEX idx_file_permissions_user_id ON file_permissions(user_id);
CREATE INDEX idx_file_permissions_role_id ON file_permissions(role_id);
```

## MongoDB Collections

### Log Data

#### system_logs
```javascript
// MongoDB collection for system logs
{
  _id: ObjectId,
  timestamp: ISODate,
  level: String, // "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
  logger: String, // Logger name/module
  message: String,
  source: {
    module: String,
    function: String,
    line_number: Number,
    file_path: String
  },
  context: {
    user_id: String,
    session_id: String,
    request_id: String,
    task_id: String
  },
  metadata: {}, // Flexible metadata object
  tags: [String],
  structured_data: {}, // Structured log data
  stack_trace: String, // For error logs
  created_at: ISODate,
  
  // Indexes
  indexes: [
    { timestamp: -1 },
    { level: 1, timestamp: -1 },
    { "context.user_id": 1, timestamp: -1 },
    { "context.task_id": 1, timestamp: -1 },
    { tags: 1, timestamp: -1 },
    { message: "text" } // Text index for full-text search
  ]
}
```

#### audit_logs
```javascript
// Security and audit event logs
{
  _id: ObjectId,
  event_type: String, // "authentication", "authorization", "data_access", etc.
  actor: {
    user_id: String,
    username: String,
    ip_address: String,
    user_agent: String
  },
  target: {
    resource_type: String, // "user", "file", "task", "system"
    resource_id: String,
    resource_name: String
  },
  action: String, // "create", "read", "update", "delete", "execute"
  outcome: String, // "success", "failure", "denied"
  severity: String, // "low", "medium", "high", "critical"
  details: {
    old_values: {},
    new_values: {},
    additional_info: {}
  },
  risk_score: Number, // 0-100
  geolocation: {
    country: String,
    region: String,
    city: String,
    latitude: Number,
    longitude: Number
  },
  timestamp: ISODate,
  retention_until: ISODate,
  
  indexes: [
    { timestamp: -1 },
    { event_type: 1, timestamp: -1 },
    { "actor.user_id": 1, timestamp: -1 },
    { "target.resource_type": 1, "target.resource_id": 1 },
    { outcome: 1, timestamp: -1 },
    { severity: 1, timestamp: -1 },
    { risk_score: -1, timestamp: -1 }
  ]
}
```

### AI and Machine Learning Data

#### ai_models
```javascript
// AI model metadata and configuration
{
  _id: ObjectId,
  name: String,
  version: String,
  model_type: String, // "language_model", "classification", "regression", etc.
  framework: String, // "tensorflow", "pytorch", "huggingface", etc.
  architecture: String,
  parameters: {
    total_params: Number,
    trainable_params: Number,
    model_size_mb: Number
  },
  configuration: {
    max_length: Number,
    temperature: Number,
    top_p: Number,
    top_k: Number,
    custom_params: {}
  },
  performance_metrics: {
    accuracy: Number,
    precision: Number,
    recall: Number,
    f1_score: Number,
    inference_time_ms: Number,
    throughput_requests_per_second: Number
  },
  training_info: {
    dataset_name: String,
    training_date: ISODate,
    epochs: Number,
    learning_rate: Number,
    batch_size: Number
  },
  deployment: {
    status: String, // "training", "available", "deprecated", "error"
    device: String, // "cpu", "gpu", "tpu"
    memory_usage_mb: Number,
    load_time_ms: Number
  },
  usage_stats: {
    total_requests: Number,
    successful_requests: Number,
    failed_requests: Number,
    avg_response_time_ms: Number,
    last_used: ISODate
  },
  created_at: ISODate,
  updated_at: ISODate,
  
  indexes: [
    { name: 1, version: 1 },
    { model_type: 1 },
    { "deployment.status": 1 },
    { "usage_stats.last_used": -1 },
    { created_at: -1 }
  ]
}
```

#### conversation_history
```javascript
// Chat conversation history
{
  _id: ObjectId,
  session_id: String,
  user_id: String,
  conversation_id: String,
  messages: [
    {
      message_id: String,
      role: String, // "user", "assistant", "system"
      content: String,
      timestamp: ISODate,
      metadata: {
        model_used: String,
        tokens_used: Number,
        processing_time_ms: Number,
        confidence_score: Number
      },
      attachments: [
        {
          type: String, // "file", "image", "code"
          content: String,
          metadata: {}
        }
      ]
    }
  ],
  context: {
    system_info: {},
    user_preferences: {},
    session_context: {}
  },
  tags: [String],
  total_tokens: Number,
  total_cost: Number,
  created_at: ISODate,
  last_activity: ISODate,
  archived: Boolean,
  
  indexes: [
    { session_id: 1 },
    { user_id: 1, last_activity: -1 },
    { conversation_id: 1 },
    { last_activity: -1 },
    { tags: 1 },
    { "messages.timestamp": -1 }
  ]
}
```

## Redis Data Structures

### Caching Layer

```redis
# User session cache
HASH user:session:{session_id}
  user_id: {user_id}
  username: {username}
  roles: {json_encoded_roles}
  permissions: {json_encoded_permissions}
  last_activity: {timestamp}
  expires_at: {timestamp}

# API rate limiting
ZSET rate_limit:{api_key}:{window}
  score: {timestamp}
  member: {request_id}

# System metrics cache
HASH metrics:current
  cpu_usage: {percentage}
  memory_usage: {percentage}
  disk_usage: {percentage}
  active_tasks: {count}
  last_updated: {timestamp}

# Task queue
LIST task_queue:priority:{priority}
  {task_id}:{execution_id}

# Real-time notifications
STREAM notifications:{user_id}
  id: {auto_generated}
  type: {notification_type}
  title: {notification_title}
  message: {notification_message}
  data: {json_encoded_data}
  timestamp: {timestamp}

# File upload progress
HASH upload:{upload_id}
  filename: {filename}
  total_size: {bytes}
  uploaded_size: {bytes}
  status: {status}
  started_at: {timestamp}
```

## TimescaleDB Schema

### Metrics and Time-Series Data

```sql
-- Create hypertable for metrics
CREATE TABLE metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('metrics', 'time');

-- Create indexes for efficient querying
CREATE INDEX idx_metrics_name_time ON metrics (metric_name, time DESC);
CREATE INDEX idx_metrics_tags ON metrics USING GIN (tags);

-- System performance metrics
CREATE TABLE system_metrics (
    time TIMESTAMPTZ NOT NULL,
    node_id VARCHAR(50) NOT NULL,
    cpu_usage DOUBLE PRECISION,
    memory_usage DOUBLE PRECISION,
    memory_total BIGINT,
    disk_usage DOUBLE PRECISION,
    disk_total BIGINT,
    network_bytes_in BIGINT,
    network_bytes_out BIGINT,
    load_average DOUBLE PRECISION,
    uptime_seconds BIGINT
);

SELECT create_hypertable('system_metrics', 'time');

-- AI model performance metrics
CREATE TABLE ai_metrics (
    time TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    request_count INTEGER DEFAULT 0,
    response_time_ms DOUBLE PRECISION,
    tokens_processed INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    memory_usage_mb DOUBLE PRECISION,
    gpu_utilization DOUBLE PRECISION
);

SELECT create_hypertable('ai_metrics', 'time');

-- Data retention policies
SELECT add_retention_policy('metrics', INTERVAL '90 days');
SELECT add_retention_policy('system_metrics', INTERVAL '1 year');
SELECT add_retention_policy('ai_metrics', INTERVAL '6 months');
```

## Database Relationships and Constraints

### Foreign Key Relationships

```sql
-- User-related relationships
ALTER TABLE user_sessions 
  ADD CONSTRAINT fk_user_sessions_user_id 
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE user_roles 
  ADD CONSTRAINT fk_user_roles_user_id 
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE user_roles 
  ADD CONSTRAINT fk_user_roles_role_id 
  FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE;

-- Task-related relationships
ALTER TABLE task_executions 
  ADD CONSTRAINT fk_task_executions_task_id 
  FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE;

ALTER TABLE task_dependencies 
  ADD CONSTRAINT fk_task_dependencies_parent 
  FOREIGN KEY (parent_task_id) REFERENCES tasks(id) ON DELETE CASCADE;

ALTER TABLE task_dependencies 
  ADD CONSTRAINT fk_task_dependencies_dependent 
  FOREIGN KEY (dependent_task_id) REFERENCES tasks(id) ON DELETE CASCADE;

-- File-related relationships
ALTER TABLE file_permissions 
  ADD CONSTRAINT fk_file_permissions_file_id 
  FOREIGN KEY (file_id) REFERENCES file_metadata(id) ON DELETE CASCADE;

ALTER TABLE file_permissions 
  ADD CONSTRAINT fk_file_permissions_user_id 
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
```

### Database Triggers and Functions

```sql
-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to tables with updated_at columns
CREATE TRIGGER update_users_updated_at 
  BEFORE UPDATE ON users 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at 
  BEFORE UPDATE ON tasks 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Audit trigger for sensitive tables
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_logs (
        table_name, 
        operation, 
        old_values, 
        new_values, 
        user_id, 
        timestamp
    ) VALUES (
        TG_TABLE_NAME,
        TG_OP,
        CASE WHEN TG_OP = 'DELETE' THEN row_to_json(OLD) ELSE NULL END,
        CASE WHEN TG_OP != 'DELETE' THEN row_to_json(NEW) ELSE NULL END,
        current_setting('app.current_user_id', true)::UUID,
        CURRENT_TIMESTAMP
    );
    
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers to sensitive tables
CREATE TRIGGER audit_users_trigger
  AFTER INSERT OR UPDATE OR DELETE ON users
  FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
```

This comprehensive database schema provides the foundation for NEO's robust data management, ensuring data integrity, performance, and scalability across all system components.
