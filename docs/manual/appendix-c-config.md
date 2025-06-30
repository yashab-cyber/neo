# Appendix C: Configuration Files
**Complete Configuration Reference and Examples**

---

## C.1 Configuration Overview

NEO's configuration system provides fine-grained control over all aspects of the system. Configuration files use YAML format for readability and support environment variable substitution, includes, and validation.

### Configuration Hierarchy
```
1. Default configuration (embedded)
2. System-wide configuration (/etc/neo/config.yml)
3. User configuration (~/.neo/config.yml)
4. Local configuration (./neo.yml)
5. Environment variables (NEO_*)
6. Command-line arguments
```

### Configuration File Locations
```bash
# System-wide (Linux/macOS)
/etc/neo/
├── config.yml
├── security.yml
├── logging.yml
└── modules/

# User-specific
~/.neo/
├── config.yml
├── profiles/
├── certificates/
└── cache/

# Windows system-wide
C:\ProgramData\NEO\
├── config.yml
├── security.yml
└── modules\

# Windows user-specific
%APPDATA%\NEO\
├── config.yml
├── profiles\
└── cache\
```

## C.2 Main Configuration File (config.yml)

### Complete Example Configuration
```yaml
# NEO Main Configuration File
# Version: 1.0
# Last Updated: 2025-06-29

# Global Settings
global:
  version: "1.0.0"
  environment: "production"  # development, staging, production
  debug: false
  log_level: "info"         # debug, info, warn, error, fatal
  timezone: "UTC"
  language: "en-US"
  
  # Data directories
  data_dir: "${NEO_DATA_DIR:/var/lib/neo}"
  log_dir: "${NEO_LOG_DIR:/var/log/neo}"
  cache_dir: "${NEO_CACHE_DIR:/var/cache/neo}"
  temp_dir: "${NEO_TEMP_DIR:/tmp/neo}"

# System Configuration
system:
  # Resource limits
  resources:
    max_cpu_percent: 80
    max_memory_mb: 8192
    max_disk_usage_percent: 90
    max_concurrent_tasks: 50
    
  # Performance settings
  performance:
    cache_size_mb: 512
    worker_threads: 8
    io_threads: 4
    network_timeout: 30
    
  # Hardware acceleration
  hardware:
    enable_gpu: true
    gpu_memory_limit_mb: 4096
    enable_vectorization: true
    enable_parallel_processing: true

# AI Engine Configuration
ai_engine:
  # Core AI settings
  core:
    model_path: "${NEO_MODELS_DIR:/var/lib/neo/models}"
    enable_deep_learning: true
    enable_neuro_learning: true
    enable_recursive_learning: true
    
  # Learning parameters
  learning:
    adaptation_rate: 0.1
    memory_consolidation: true
    context_window_size: 4096
    max_response_length: 2048
    
  # Model configuration
  models:
    default_language_model: "neo-gpt-v1"
    default_vision_model: "neo-vision-v1"
    default_reasoning_model: "neo-reason-v1"
    
  # Inference settings
  inference:
    batch_size: 32
    max_sequence_length: 1024
    temperature: 0.7
    top_p: 0.9
    top_k: 50

# Security Configuration
security:
  # Authentication
  authentication:
    method: "multi_factor"        # simple, multi_factor, enterprise
    session_timeout: 3600         # seconds
    max_failed_attempts: 5
    lockout_duration: 300         # seconds
    
  # Encryption
  encryption:
    algorithm: "AES-256-GCM"
    key_rotation_days: 90
    encrypt_data_at_rest: true
    encrypt_data_in_transit: true
    
  # Access control
  access_control:
    enable_rbac: true
    default_permissions: "read"
    admin_approval_required: true
    audit_all_actions: true
    
  # Threat detection
  threat_detection:
    enable_real_time_monitoring: true
    anomaly_detection_threshold: 0.8
    auto_quarantine: true
    alert_threshold: "medium"

# Network Configuration
network:
  # Server settings
  server:
    host: "0.0.0.0"
    port: 8443
    enable_ssl: true
    ssl_cert_path: "/etc/neo/certs/server.crt"
    ssl_key_path: "/etc/neo/certs/server.key"
    
  # Client settings
  client:
    connect_timeout: 10
    read_timeout: 30
    max_retries: 3
    retry_delay: 1
    
  # Proxy settings
  proxy:
    enable: false
    http_proxy: ""
    https_proxy: ""
    no_proxy: "localhost,127.0.0.1"

# Database Configuration
database:
  # Primary database
  primary:
    type: "postgresql"
    host: "${DB_HOST:localhost}"
    port: "${DB_PORT:5432}"
    database: "${DB_NAME:neo}"
    username: "${DB_USER:neo}"
    password: "${DB_PASSWORD}"
    
  # Connection pool
  pool:
    min_connections: 5
    max_connections: 50
    idle_timeout: 300
    max_lifetime: 3600
    
  # Caching database
  cache:
    type: "redis"
    host: "${REDIS_HOST:localhost}"
    port: "${REDIS_PORT:6379}"
    password: "${REDIS_PASSWORD}"
    database: 0

# Logging Configuration
logging:
  # Log levels per component
  levels:
    root: "info"
    neo.ai: "debug"
    neo.security: "warn"
    neo.system: "info"
    neo.network: "error"
    
  # Output configuration
  outputs:
    - type: "file"
      path: "${NEO_LOG_DIR}/neo.log"
      format: "json"
      rotation: "daily"
      max_size: "100MB"
      max_age: "30d"
      
    - type: "console"
      format: "text"
      color: true
      
    - type: "syslog"
      network: "udp"
      address: "localhost:514"
      facility: "local0"

# Module Configuration
modules:
  # Core modules (always enabled)
  core:
    - "ai_engine"
    - "security"
    - "system_control"
    - "learning"
    
  # Optional modules
  optional:
    cybersecurity:
      enabled: true
      config_file: "modules/cybersecurity.yml"
      
    penetration_testing:
      enabled: true
      config_file: "modules/pentest.yml"
      
    code_development:
      enabled: true
      config_file: "modules/development.yml"
      
    research:
      enabled: true
      config_file: "modules/research.yml"

# Monitoring Configuration
monitoring:
  # Metrics collection
  metrics:
    enabled: true
    interval: 60              # seconds
    retention: "7d"
    
  # Health checks
  health:
    enabled: true
    interval: 30              # seconds
    timeout: 5                # seconds
    
  # Alerting
  alerts:
    enabled: true
    smtp_server: "${SMTP_SERVER}"
    smtp_port: "${SMTP_PORT:587}"
    smtp_username: "${SMTP_USER}"
    smtp_password: "${SMTP_PASSWORD}"
    
# Backup Configuration
backup:
  # Automatic backup
  automatic:
    enabled: true
    schedule: "0 2 * * *"      # Daily at 2 AM (cron format)
    retention_days: 30
    compression: true
    
  # Backup locations
  locations:
    local:
      path: "${BACKUP_DIR:/var/backups/neo}"
      enabled: true
      
    cloud:
      provider: "aws_s3"       # aws_s3, azure_blob, gcp_storage
      bucket: "${BACKUP_BUCKET}"
      enabled: false

# Integration Configuration
integrations:
  # External APIs
  apis:
    openai:
      enabled: false
      api_key: "${OPENAI_API_KEY}"
      
    google_cloud:
      enabled: false
      credentials_file: "/etc/neo/gcp-credentials.json"
      
    aws:
      enabled: false
      access_key: "${AWS_ACCESS_KEY}"
      secret_key: "${AWS_SECRET_KEY}"
      region: "${AWS_REGION:us-east-1}"
      
  # Third-party tools
  tools:
    docker:
      enabled: true
      socket_path: "/var/run/docker.sock"
      
    kubernetes:
      enabled: false
      config_file: "~/.kube/config"
      
    git:
      enabled: true
      default_branch: "main"
```

## C.3 Security Configuration (security.yml)

```yaml
# NEO Security Configuration
# Comprehensive security settings

# Authentication Configuration
authentication:
  # Multi-factor authentication
  mfa:
    enabled: true
    methods:
      - "totp"              # Time-based One-Time Password
      - "sms"               # SMS verification
      - "email"             # Email verification
      - "hardware_token"    # Hardware security keys
    grace_period: 300       # seconds
    
  # Password policy
  password_policy:
    min_length: 12
    require_uppercase: true
    require_lowercase: true
    require_numbers: true
    require_symbols: true
    max_age_days: 90
    history_count: 12       # Remember last N passwords
    
  # Session management
  sessions:
    timeout_idle: 1800      # 30 minutes
    timeout_absolute: 28800 # 8 hours
    concurrent_limit: 3     # Max concurrent sessions per user
    secure_cookies: true
    samesite_policy: "strict"

# Authorization Configuration
authorization:
  # Role-Based Access Control
  rbac:
    enabled: true
    roles:
      admin:
        permissions:
          - "system:*"
          - "security:*"
          - "users:*"
          
      security_analyst:
        permissions:
          - "security:read"
          - "security:scan"
          - "security:investigate"
          - "incidents:*"
          
      developer:
        permissions:
          - "development:*"
          - "system:read"
          - "code:*"
          
      user:
        permissions:
          - "ai:query"
          - "system:read"
          - "files:own"
          
  # Attribute-Based Access Control
  abac:
    enabled: true
    policies_file: "/etc/neo/security/abac-policies.json"

# Encryption Configuration
encryption:
  # Data at rest
  at_rest:
    algorithm: "AES-256-GCM"
    key_derivation: "PBKDF2"
    iterations: 100000
    
  # Data in transit
  in_transit:
    min_tls_version: "1.3"
    cipher_suites:
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_CHACHA20_POLY1305_SHA256"
      - "TLS_AES_128_GCM_SHA256"
    certificate_pinning: true
    
  # Key management
  key_management:
    provider: "internal"      # internal, aws_kms, azure_keyvault, hsm
    rotation_schedule: "0 0 1 * *"  # Monthly rotation
    backup_keys: true

# Audit Configuration
audit:
  enabled: true
  
  # Audit events
  events:
    authentication: true
    authorization: true
    data_access: true
    configuration_changes: true
    security_events: true
    system_events: true
    
  # Audit storage
  storage:
    type: "database"          # database, file, syslog, splunk
    retention_days: 365
    compression: true
    encryption: true
    
  # Audit alerts
  alerts:
    failed_logins_threshold: 5
    privilege_escalation: true
    unusual_access_patterns: true
    data_exfiltration_attempts: true

# Threat Detection Configuration
threat_detection:
  # Behavioral analysis
  behavioral:
    enabled: true
    learning_period_days: 30
    sensitivity: "medium"     # low, medium, high
    
  # Signature-based detection
  signatures:
    enabled: true
    update_interval: 3600     # seconds
    sources:
      - "internal"
      - "commercial_feeds"
      - "open_source"
      
  # Anomaly detection
  anomaly:
    enabled: true
    algorithms:
      - "statistical"
      - "machine_learning"
      - "rule_based"
    threshold: 0.8
    
  # Response actions
  responses:
    auto_block: true
    quarantine_suspicious: true
    alert_administrators: true
    log_all_events: true

# Network Security
network_security:
  # Firewall configuration
  firewall:
    enabled: true
    default_policy: "deny"
    
    rules:
      - name: "ssh_access"
        action: "allow"
        protocol: "tcp"
        port: 22
        source: "admin_networks"
        
      - name: "https_api"
        action: "allow"
        protocol: "tcp"
        port: 8443
        source: "any"
        
      - name: "monitoring"
        action: "allow"
        protocol: "tcp"
        port: 9090
        source: "monitoring_networks"
        
  # Intrusion detection
  ids:
    enabled: true
    mode: "inline"            # inline, passive
    sensitivity: "medium"
    
  # DDoS protection
  ddos_protection:
    enabled: true
    rate_limiting:
      requests_per_minute: 1000
      requests_per_hour: 10000
    connection_limiting:
      max_connections: 1000
      max_connections_per_ip: 10

# Compliance Configuration
compliance:
  # Regulatory frameworks
  frameworks:
    gdpr:
      enabled: true
      data_retention_days: 365
      consent_tracking: true
      
    hipaa:
      enabled: false
      encryption_required: true
      audit_logs_required: true
      
    pci_dss:
      enabled: false
      cardholder_data_protection: true
      
    sox:
      enabled: false
      financial_controls: true
      
  # Privacy controls
  privacy:
    data_minimization: true
    purpose_limitation: true
    anonymization: true
    right_to_erasure: true

# Security Monitoring
monitoring:
  # SIEM integration
  siem:
    enabled: false
    type: "splunk"            # splunk, elastic, qradar, sentinel
    endpoint: "${SIEM_ENDPOINT}"
    api_key: "${SIEM_API_KEY}"
    
  # Security metrics
  metrics:
    enabled: true
    collection_interval: 60   # seconds
    metrics:
      - "authentication_events"
      - "authorization_failures"
      - "threat_detections"
      - "security_alerts"
      - "vulnerability_counts"
      
  # Alerting
  alerts:
    channels:
      email:
        enabled: true
        recipients:
          - "security-team@company.com"
        severity_threshold: "medium"
        
      slack:
        enabled: true
        webhook_url: "${SLACK_WEBHOOK}"
        severity_threshold: "high"
        
      pagerduty:
        enabled: false
        service_key: "${PAGERDUTY_KEY}"
        severity_threshold: "critical"

# Incident Response
incident_response:
  # Automated response
  automated:
    enabled: true
    response_time_seconds: 30
    
    actions:
      quarantine:
        enabled: true
        threshold: "high"
        
      block_ip:
        enabled: true
        threshold: "critical"
        duration: 3600          # seconds
        
      disable_account:
        enabled: true
        threshold: "critical"
        
  # Manual response
  manual:
    escalation_time: 1800     # 30 minutes
    notification_channels:
      - "email"
      - "sms"
      - "slack"
      
  # Evidence collection
  evidence:
    auto_collect: true
    retention_days: 90
    types:
      - "logs"
      - "network_traffic"
      - "memory_dumps"
      - "file_samples"

# Security Training
training:
  # Awareness training
  awareness:
    enabled: true
    frequency_days: 90
    topics:
      - "phishing"
      - "social_engineering"
      - "password_security"
      - "data_protection"
      
  # Simulation exercises
  simulations:
    phishing:
      enabled: true
      frequency_days: 30
      
    incident_response:
      enabled: true
      frequency_days: 90
```

## C.4 Module Configuration Examples

### Cybersecurity Module (modules/cybersecurity.yml)
```yaml
# Cybersecurity Module Configuration

module:
  name: "cybersecurity"
  version: "1.0.0"
  enabled: true

# Threat Intelligence
threat_intelligence:
  feeds:
    commercial:
      - name: "CrowdStrike"
        enabled: false
        api_key: "${CROWDSTRIKE_API_KEY}"
        
    open_source:
      - name: "MISP"
        enabled: true
        url: "${MISP_URL}"
        api_key: "${MISP_API_KEY}"
        
      - name: "OTX"
        enabled: true
        api_key: "${OTX_API_KEY}"
        
  processing:
    auto_import: true
    confidence_threshold: 0.7
    false_positive_learning: true

# Vulnerability Management
vulnerability_management:
  scanners:
    nessus:
      enabled: false
      host: "${NESSUS_HOST}"
      access_key: "${NESSUS_ACCESS_KEY}"
      secret_key: "${NESSUS_SECRET_KEY}"
      
    openvas:
      enabled: true
      host: "localhost"
      port: 9392
      username: "${OPENVAS_USER}"
      password: "${OPENVAS_PASSWORD}"
      
  scanning:
    schedule: "0 2 * * 0"      # Weekly on Sunday at 2 AM
    scan_profiles:
      - "full_and_fast"
      - "discovery"
    auto_remediation: false

# Security Operations Center
soc:
  enabled: true
  
  # Incident classification
  classification:
    auto_classify: true
    severity_matrix:
      - condition: "malware_detected"
        severity: "high"
      - condition: "data_breach"
        severity: "critical"
      - condition: "unauthorized_access"
        severity: "medium"
        
  # Response automation
  automation:
    playbooks_dir: "/etc/neo/playbooks"
    auto_execute: true
    approval_required: false
```

### Development Module (modules/development.yml)
```yaml
# Development Module Configuration

module:
  name: "development"
  version: "1.0.0"
  enabled: true

# Code Analysis
code_analysis:
  # Static analysis tools
  static_analysis:
    sonarqube:
      enabled: false
      url: "${SONARQUBE_URL}"
      token: "${SONARQUBE_TOKEN}"
      
    eslint:
      enabled: true
      config_file: ".eslintrc.json"
      
    pylint:
      enabled: true
      config_file: ".pylintrc"
      
  # Security scanning
  security_scanning:
    snyk:
      enabled: false
      api_token: "${SNYK_TOKEN}"
      
    bandit:
      enabled: true
      config_file: ".bandit"
      
    semgrep:
      enabled: true
      rules: "auto"

# CI/CD Integration
cicd:
  platforms:
    jenkins:
      enabled: false
      url: "${JENKINS_URL}"
      username: "${JENKINS_USER}"
      api_token: "${JENKINS_TOKEN}"
      
    github_actions:
      enabled: true
      token: "${GITHUB_TOKEN}"
      
    gitlab_ci:
      enabled: false
      url: "${GITLAB_URL}"
      token: "${GITLAB_TOKEN}"
      
  # Quality gates
  quality_gates:
    code_coverage_threshold: 80
    security_vulnerability_threshold: 0
    code_quality_threshold: "A"
    test_success_rate: 100

# Development Environment
development_environment:
  # Container support
  containers:
    docker:
      enabled: true
      socket: "/var/run/docker.sock"
      
    podman:
      enabled: false
      socket: "/run/podman/podman.sock"
      
  # Language support
  languages:
    python:
      versions: ["3.8", "3.9", "3.10", "3.11"]
      package_manager: "pip"
      
    javascript:
      versions: ["16", "18", "20"]
      package_manager: "npm"
      
    java:
      versions: ["11", "17", "21"]
      build_tool: "maven"
      
  # IDE integration
  ide_integration:
    vscode:
      enabled: true
      extensions_auto_install: true
      
    intellij:
      enabled: true
      plugins_auto_install: false
```

## C.5 Environment Variables

### Core Environment Variables
```bash
# Core NEO settings
export NEO_HOME="/opt/neo"
export NEO_CONFIG_DIR="/etc/neo"
export NEO_DATA_DIR="/var/lib/neo"
export NEO_LOG_DIR="/var/log/neo"
export NEO_CACHE_DIR="/var/cache/neo"
export NEO_TEMP_DIR="/tmp/neo"

# Database settings
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="neo"
export DB_USER="neo"
export DB_PASSWORD="secure_password"

# Redis settings
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_PASSWORD="redis_password"

# Security settings
export NEO_SECRET_KEY="your-secret-key-here"
export NEO_ENCRYPTION_KEY="your-encryption-key"
export SSL_CERT_PATH="/etc/neo/certs/server.crt"
export SSL_KEY_PATH="/etc/neo/certs/server.key"

# API keys for external services
export OPENAI_API_KEY="your-openai-key"
export CROWDSTRIKE_API_KEY="your-crowdstrike-key"
export GITHUB_TOKEN="your-github-token"

# Monitoring and alerting
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="alerts@company.com"
export SMTP_PASSWORD="email_password"
export SLACK_WEBHOOK="https://hooks.slack.com/services/..."

# Cloud provider settings
export AWS_ACCESS_KEY="your-aws-access-key"
export AWS_SECRET_KEY="your-aws-secret-key"
export AWS_REGION="us-east-1"
export BACKUP_BUCKET="neo-backups"

# Performance tuning
export NEO_MAX_WORKERS="8"
export NEO_MEMORY_LIMIT="8192"
export NEO_CPU_LIMIT="80"
```

## C.6 Configuration Validation

### Validation Script
```bash
#!/bin/bash
# NEO Configuration Validation Script

echo "Validating NEO configuration..."

# Check configuration file syntax
neo config validate --file /etc/neo/config.yml
if [ $? -ne 0 ]; then
    echo "❌ Configuration file has syntax errors"
    exit 1
fi

# Check required environment variables
required_vars=(
    "NEO_HOME"
    "DB_HOST"
    "DB_PASSWORD"
    "NEO_SECRET_KEY"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set"
        exit 1
    fi
done

# Test database connectivity
neo config test-db
if [ $? -ne 0 ]; then
    echo "❌ Database connection failed"
    exit 1
fi

# Test security configuration
neo config test-security
if [ $? -ne 0 ]; then
    echo "❌ Security configuration invalid"
    exit 1
fi

echo "✅ Configuration validation passed"
```

---

**Next**: [Appendix D: Glossary](appendix-d-glossary.md)

*Proper configuration is the foundation of a secure and efficient NEO deployment. These templates provide a solid starting point for any environment.*
