# Chapter 24: Integration Setup

## Overview

NEO's integration system allows seamless connectivity with external services, APIs, databases, and third-party applications. This chapter provides comprehensive guidance on setting up and managing integrations for maximum productivity and automation.

## Core Integration Types

### 1. API Integrations

```python
# REST API Integration Example
from neo.integrations import APIConnector

# Configure API connection
api_config = {
    "name": "company_crm",
    "base_url": "https://api.company.com/v2",
    "authentication": {
        "type": "oauth2",
        "client_id": "your_client_id",
        "client_secret": "your_client_secret",
        "scope": ["read", "write"]
    },
    "rate_limit": {
        "requests_per_minute": 100,
        "retry_on_limit": True
    }
}

# Create connector
crm_api = APIConnector(api_config)

# Use in NEO commands
neo.integrate.register("crm", crm_api)
```

### 2. Database Connections

```yaml
# database_config.yaml
databases:
  production:
    type: "postgresql"
    host: "prod-db.company.com"
    port: 5432
    database: "main_db"
    username: "neo_user"
    password_env: "NEO_DB_PASSWORD"
    ssl_mode: "require"
    connection_pool:
      min_connections: 5
      max_connections: 20
      
  analytics:
    type: "mongodb"
    host: "analytics.company.com"
    port: 27017
    database: "analytics"
    authentication:
      mechanism: "SCRAM-SHA-256"
      username: "analytics_user"
      password_env: "NEO_MONGO_PASSWORD"
```

### 3. Cloud Service Integrations

```python
# Cloud service configuration
cloud_integrations = {
    "aws": {
        "access_key_id": "env:AWS_ACCESS_KEY_ID",
        "secret_access_key": "env:AWS_SECRET_ACCESS_KEY",
        "region": "us-west-2",
        "services": ["s3", "ec2", "lambda", "rds"]
    },
    "google_cloud": {
        "service_account_key": "/config/gcp-service-account.json",
        "project_id": "my-project-123",
        "services": ["storage", "compute", "bigquery"]
    },
    "azure": {
        "tenant_id": "env:AZURE_TENANT_ID",
        "client_id": "env:AZURE_CLIENT_ID",
        "client_secret": "env:AZURE_CLIENT_SECRET",
        "services": ["storage", "compute", "cognitive_services"]
    }
}
```

## Communication Platform Integrations

### Slack Integration

```python
# Slack bot setup
slack_config = {
    "bot_token": "env:SLACK_BOT_TOKEN",
    "app_token": "env:SLACK_APP_TOKEN",
    "signing_secret": "env:SLACK_SIGNING_SECRET",
    "channels": {
        "alerts": "#neo-alerts",
        "reports": "#daily-reports",
        "general": "#neo-general"
    }
}

# Register Slack commands
@neo.slack.command("/neo-status")
def neo_status_command(ack, body, client):
    ack()
    status = neo.system.get_status()
    client.chat_postMessage(
        channel=body["channel_id"],
        text=f"NEO Status: {status.summary}"
    )

@neo.slack.command("/neo-task")
def neo_task_command(ack, body, client):
    ack()
    task = body["text"]
    result = neo.execute(task)
    client.chat_postMessage(
        channel=body["channel_id"],
        text=f"Task completed: {result.summary}"
    )
```

### Microsoft Teams Integration

```yaml
# teams_config.yaml
microsoft_teams:
  app_id: "env:TEAMS_APP_ID"
  app_password: "env:TEAMS_APP_PASSWORD"
  tenant_id: "env:TEAMS_TENANT_ID"
  
  bot_settings:
    name: "NEO Assistant"
    description: "AI Assistant for task automation"
    
  webhooks:
    alerts: "env:TEAMS_ALERTS_WEBHOOK"
    reports: "env:TEAMS_REPORTS_WEBHOOK"
    
  commands:
    - name: "status"
      description: "Get NEO system status"
    - name: "execute"
      description: "Execute a NEO command"
    - name: "schedule"
      description: "Schedule a task"
```

### Discord Integration

```python
# Discord bot integration
import discord
from neo.integrations.discord import DiscordBot

class NEODiscordBot(DiscordBot):
    def __init__(self):
        super().__init__(token="env:DISCORD_BOT_TOKEN")
        
    @discord.slash_command(description="Get system information")
    async def sysinfo(self, ctx):
        info = neo.system.get_info()
        embed = discord.Embed(title="System Information")
        embed.add_field(name="CPU Usage", value=f"{info.cpu_usage}%")
        embed.add_field(name="Memory Usage", value=f"{info.memory_usage}%")
        embed.add_field(name="Disk Usage", value=f"{info.disk_usage}%")
        await ctx.respond(embed=embed)
        
    @discord.slash_command(description="Execute NEO command")
    async def neo(self, ctx, command: str):
        try:
            result = neo.execute(command)
            await ctx.respond(f"✅ {result.message}")
        except Exception as e:
            await ctx.respond(f"❌ Error: {str(e)}")
```

## Development Tool Integrations

### Git Integration

```python
# Git repository integration
git_config = {
    "repositories": {
        "main_project": {
            "url": "https://github.com/company/main-project.git",
            "local_path": "/workspace/main-project",
            "branch": "main",
            "auto_pull": True,
            "hooks": {
                "pre_commit": "neo.code.lint",
                "post_commit": "neo.code.test",
                "pre_push": "neo.code.security_scan"
            }
        }
    },
    "automation": {
        "auto_commit_messages": True,
        "conventional_commits": True,
        "auto_branch_naming": True
    }
}

# Git automation commands
@neo.command
def git_smart_commit(message=""):
    """Intelligent git commit with automated checks"""
    # Run pre-commit checks
    lint_result = neo.code.lint()
    test_result = neo.code.test()
    
    if not lint_result.passed:
        return f"❌ Lint errors found: {lint_result.errors}"
    
    if not test_result.passed:
        return f"❌ Tests failed: {test_result.failures}"
    
    # Generate commit message if not provided
    if not message:
        changes = neo.git.get_changes()
        message = neo.ai.generate_commit_message(changes)
    
    # Commit and push
    neo.git.add_all()
    neo.git.commit(message)
    neo.git.push()
    
    return f"✅ Committed and pushed: {message}"
```

### Jenkins Integration

```yaml
# jenkins_config.yaml
jenkins:
  url: "https://jenkins.company.com"
  username: "neo_user"
  api_token: "env:JENKINS_API_TOKEN"
  
  jobs:
    - name: "deploy_production"
      parameters:
        - name: "branch"
          default: "main"
        - name: "environment"
          default: "production"
          
    - name: "run_tests"
      trigger_on:
        - "code_change"
        - "pull_request"
        
  automation:
    auto_trigger_on_commit: true
    notify_on_failure: true
    retry_failed_builds: 3
```

### Docker Integration

```python
# Docker container management
docker_config = {
    "registries": {
        "docker_hub": {
            "username": "env:DOCKER_USERNAME",
            "password": "env:DOCKER_PASSWORD"
        },
        "private_registry": {
            "url": "registry.company.com",
            "username": "env:PRIVATE_REGISTRY_USER",
            "password": "env:PRIVATE_REGISTRY_PASS"
        }
    },
    "containers": {
        "web_app": {
            "image": "company/webapp:latest",
            "ports": ["8080:80"],
            "environment": {
                "NODE_ENV": "production",
                "DATABASE_URL": "env:DATABASE_URL"
            },
            "volumes": [
                "/app/data:/data",
                "/app/logs:/logs"
            ]
        }
    }
}

# Docker automation commands
@neo.command
def docker_deploy(service_name, tag="latest"):
    """Deploy Docker service with automated checks"""
    # Pull latest image
    neo.docker.pull(f"{service_name}:{tag}")
    
    # Run security scan
    scan_result = neo.security.scan_image(f"{service_name}:{tag}")
    if scan_result.high_vulnerabilities > 0:
        return f"❌ Security vulnerabilities found: {scan_result.high_vulnerabilities}"
    
    # Deploy with rolling update
    neo.docker.deploy(service_name, tag, strategy="rolling_update")
    
    # Health check
    health = neo.docker.health_check(service_name, timeout=60)
    if not health.healthy:
        neo.docker.rollback(service_name)
        return f"❌ Deployment failed health check, rolled back"
    
    return f"✅ Successfully deployed {service_name}:{tag}"
```

## Monitoring and Analytics Integrations

### Prometheus/Grafana Integration

```yaml
# monitoring_config.yaml
prometheus:
  url: "http://prometheus.company.com:9090"
  scrape_configs:
    - job_name: "neo_metrics"
      static_configs:
        - targets: ["localhost:8080"]
      scrape_interval: "15s"
      
grafana:
  url: "http://grafana.company.com:3000"
  api_key: "env:GRAFANA_API_KEY"
  dashboards:
    - name: "NEO System Overview"
      file: "/config/grafana/neo_overview.json"
    - name: "NEO Performance Metrics"
      file: "/config/grafana/neo_performance.json"
```

### ELK Stack Integration

```python
# Elasticsearch integration
elasticsearch_config = {
    "hosts": ["elasticsearch.company.com:9200"],
    "http_auth": ("neo_user", "env:ELASTIC_PASSWORD"),
    "use_ssl": True,
    "verify_certs": True,
    "indices": {
        "logs": "neo-logs-*",
        "metrics": "neo-metrics-*",
        "events": "neo-events-*"
    }
}

# Automated log analysis
@neo.scheduled("0 */6 * * *")  # Every 6 hours
def analyze_logs():
    """Analyze logs and generate insights"""
    # Query recent logs
    logs = neo.elasticsearch.search(
        index="neo-logs-*",
        body={
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": "now-6h"
                    }
                }
            }
        }
    )
    
    # Analyze patterns
    error_patterns = neo.ai.analyze_log_patterns(logs, focus="errors")
    performance_trends = neo.ai.analyze_performance_trends(logs)
    
    # Generate report
    report = neo.report.create("log_analysis", {
        "error_patterns": error_patterns,
        "performance_trends": performance_trends,
        "recommendations": neo.ai.generate_recommendations(logs)
    })
    
    # Send to stakeholders
    neo.notify.send_report(report, recipients=["devops-team"])
```

## Enterprise System Integrations

### Active Directory Integration

```python
# Active Directory configuration
ad_config = {
    "server": "ldap://ad.company.com",
    "domain": "company.com",
    "bind_user": "CN=neo-service,OU=Service Accounts,DC=company,DC=com",
    "bind_password": "env:AD_SERVICE_PASSWORD",
    "user_base_dn": "OU=Users,DC=company,DC=com",
    "group_base_dn": "OU=Groups,DC=company,DC=com",
    "ssl": True
}

# User authentication and authorization
class ADAuthenticator:
    def __init__(self, config):
        self.config = config
        self.connection = neo.ldap.connect(config)
    
    def authenticate_user(self, username, password):
        """Authenticate user against Active Directory"""
        user_dn = f"CN={username},{self.config['user_base_dn']}"
        
        try:
            # Attempt to bind with user credentials
            result = self.connection.simple_bind_s(user_dn, password)
            return {"success": True, "user": username}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_user_groups(self, username):
        """Get user's group memberships"""
        search_filter = f"(sAMAccountName={username})"
        result = self.connection.search_s(
            self.config['user_base_dn'],
            ldap.SCOPE_SUBTREE,
            search_filter,
            ['memberOf']
        )
        return result[0][1]['memberOf'] if result else []
```

### SAP Integration

```python
# SAP system integration
sap_config = {
    "connection": {
        "ashost": "sap.company.com",
        "sysnr": "00",
        "client": "100",
        "user": "NEO_USER",
        "passwd": "env:SAP_PASSWORD"
    },
    "rfc_functions": {
        "get_employee_data": "Z_GET_EMPLOYEE_DATA",
        "create_purchase_order": "Z_CREATE_PO",
        "get_financial_data": "Z_GET_FIN_DATA"
    }
}

# SAP RFC integration
from pyrfc import Connection

class SAPConnector:
    def __init__(self, config):
        self.config = config
        self.connection = Connection(**config['connection'])
    
    def get_employee_data(self, employee_id):
        """Retrieve employee data from SAP"""
        result = self.connection.call(
            self.config['rfc_functions']['get_employee_data'],
            EMPLOYEE_ID=employee_id
        )
        return result
    
    def create_purchase_order(self, po_data):
        """Create purchase order in SAP"""
        result = self.connection.call(
            self.config['rfc_functions']['create_purchase_order'],
            **po_data
        )
        return result
```

## IoT and Hardware Integrations

### MQTT Integration

```python
# MQTT broker configuration
mqtt_config = {
    "broker": "mqtt.company.com",
    "port": 8883,
    "username": "neo_client",
    "password": "env:MQTT_PASSWORD",
    "use_ssl": True,
    "topics": {
        "sensors": "company/sensors/+/data",
        "commands": "company/devices/+/commands",
        "alerts": "company/alerts/+"
    }
}

# MQTT client for IoT integration
class IoTMQTTClient:
    def __init__(self, config):
        self.config = config
        self.client = neo.mqtt.Client()
        self.setup_connection()
        
    def setup_connection(self):
        """Setup MQTT connection with SSL and authentication"""
        self.client.username_pw_set(
            self.config['username'],
            self.config['password']
        )
        
        if self.config['use_ssl']:
            self.client.tls_set()
        
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
    def on_connect(self, client, userdata, flags, rc):
        """Callback for successful connection"""
        neo.log(f"Connected to MQTT broker with result code {rc}")
        
        # Subscribe to all configured topics
        for topic_name, topic_pattern in self.config['topics'].items():
            client.subscribe(topic_pattern)
            neo.log(f"Subscribed to {topic_pattern}")
    
    def on_message(self, client, userdata, msg):
        """Process incoming MQTT messages"""
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        
        # Route message based on topic
        if "sensors" in topic:
            self.handle_sensor_data(topic, payload)
        elif "alerts" in topic:
            self.handle_alert(topic, payload)
    
    def handle_sensor_data(self, topic, data):
        """Process sensor data"""
        device_id = topic.split('/')[2]
        
        # Store data
        neo.database.insert("sensor_data", {
            "device_id": device_id,
            "timestamp": datetime.now(),
            "data": data
        })
        
        # Check for anomalies
        if neo.ai.detect_anomaly(device_id, data):
            neo.alert.send(f"Anomaly detected on device {device_id}")
```

### Serial Device Integration

```python
# Serial device communication
import serial
import json

class SerialDeviceManager:
    def __init__(self):
        self.devices = {}
        self.scan_ports()
    
    def scan_ports(self):
        """Scan for available serial devices"""
        available_ports = neo.serial.list_ports()
        
        for port in available_ports:
            try:
                device = serial.Serial(port, 9600, timeout=1)
                device_info = self.identify_device(device)
                
                if device_info:
                    self.devices[port] = {
                        "connection": device,
                        "info": device_info,
                        "last_communication": datetime.now()
                    }
                    neo.log(f"Found device: {device_info['name']} on {port}")
                    
            except Exception as e:
                neo.log(f"Error connecting to {port}: {e}")
    
    def identify_device(self, device):
        """Identify connected device"""
        # Send identification command
        device.write(b"ID?\n")
        response = device.readline().decode().strip()
        
        # Parse device information
        if response.startswith("NEO_SENSOR"):
            return {"type": "sensor", "name": response}
        elif response.startswith("NEO_ACTUATOR"):
            return {"type": "actuator", "name": response}
        
        return None
    
    def send_command(self, port, command):
        """Send command to device"""
        if port in self.devices:
            device = self.devices[port]["connection"]
            device.write(f"{command}\n".encode())
            response = device.readline().decode().strip()
            return response
        else:
            raise Exception(f"Device not found on port {port}")
```

## Integration Security

### API Key Management

```python
# Secure API key storage and rotation
class APIKeyManager:
    def __init__(self):
        self.key_store = neo.security.EncryptedKeyStore()
        
    def store_api_key(self, service_name, api_key, metadata=None):
        """Securely store API key"""
        encrypted_key = self.key_store.encrypt(api_key)
        
        key_record = {
            "service": service_name,
            "encrypted_key": encrypted_key,
            "created_at": datetime.now(),
            "metadata": metadata or {},
            "rotation_due": datetime.now() + timedelta(days=90)
        }
        
        self.key_store.store(service_name, key_record)
        neo.audit.log("api_key_stored", {"service": service_name})
    
    def get_api_key(self, service_name):
        """Retrieve and decrypt API key"""
        key_record = self.key_store.retrieve(service_name)
        
        if not key_record:
            raise Exception(f"API key not found for service: {service_name}")
        
        # Check if rotation is due
        if datetime.now() > key_record['rotation_due']:
            neo.log.warning(f"API key rotation due for {service_name}")
        
        decrypted_key = self.key_store.decrypt(key_record['encrypted_key'])
        neo.audit.log("api_key_accessed", {"service": service_name})
        
        return decrypted_key
    
    def rotate_api_key(self, service_name, new_key):
        """Rotate API key"""
        # Store old key for rollback
        old_record = self.key_store.retrieve(service_name)
        self.key_store.store(f"{service_name}_backup", old_record)
        
        # Store new key
        self.store_api_key(service_name, new_key)
        
        neo.audit.log("api_key_rotated", {"service": service_name})
```

### OAuth 2.0 Flow

```python
# OAuth 2.0 implementation
class OAuth2Handler:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token_store = neo.security.TokenStore()
    
    def get_authorization_url(self, service, scopes):
        """Generate OAuth authorization URL"""
        state = neo.security.generate_state_token()
        
        auth_url = (
            f"https://oauth.{service}.com/authorize?"
            f"client_id={self.client_id}&"
            f"redirect_uri={self.redirect_uri}&"
            f"scope={'+'.join(scopes)}&"
            f"response_type=code&"
            f"state={state}"
        )
        
        # Store state for validation
        self.token_store.store_state(state, service)
        
        return auth_url
    
    def exchange_code_for_token(self, code, state, service):
        """Exchange authorization code for access token"""
        # Validate state
        if not self.token_store.validate_state(state, service):
            raise Exception("Invalid state parameter")
        
        # Exchange code for token
        token_response = neo.http.post(
            f"https://oauth.{service}.com/token",
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.redirect_uri
            }
        )
        
        if token_response.status_code == 200:
            token_data = token_response.json()
            
            # Store tokens securely
            self.token_store.store_tokens(service, {
                "access_token": token_data["access_token"],
                "refresh_token": token_data.get("refresh_token"),
                "expires_at": datetime.now() + timedelta(
                    seconds=token_data["expires_in"]
                )
            })
            
            return token_data
        else:
            raise Exception(f"Token exchange failed: {token_response.text}")
```

## Integration Testing

### Integration Test Framework

```python
# Comprehensive integration testing
class IntegrationTestSuite:
    def __init__(self):
        self.test_results = []
        
    def test_api_integration(self, service_name):
        """Test API integration"""
        try:
            # Test authentication
            auth_result = neo.integrations.test_auth(service_name)
            assert auth_result.success, f"Authentication failed: {auth_result.error}"
            
            # Test basic API calls
            api_result = neo.integrations.test_api_call(service_name, "GET", "/health")
            assert api_result.status_code == 200, f"API call failed: {api_result.status_code}"
            
            # Test rate limiting
            rate_limit_result = neo.integrations.test_rate_limits(service_name)
            assert rate_limit_result.respected, "Rate limits not properly handled"
            
            self.test_results.append({
                "service": service_name,
                "test": "api_integration",
                "status": "passed"
            })
            
        except Exception as e:
            self.test_results.append({
                "service": service_name,
                "test": "api_integration",
                "status": "failed",
                "error": str(e)
            })
    
    def test_database_integration(self, db_name):
        """Test database integration"""
        try:
            # Test connection
            connection = neo.database.connect(db_name)
            assert connection.is_connected(), "Database connection failed"
            
            # Test basic operations
            test_data = {"test_id": "integration_test", "timestamp": datetime.now()}
            
            # Insert test data
            insert_result = connection.insert("integration_test", test_data)
            assert insert_result.success, "Insert operation failed"
            
            # Query test data
            query_result = connection.query(
                "SELECT * FROM integration_test WHERE test_id = %s",
                ("integration_test",)
            )
            assert len(query_result) > 0, "Query operation failed"
            
            # Clean up
            connection.delete("integration_test", {"test_id": "integration_test"})
            
            self.test_results.append({
                "database": db_name,
                "test": "database_integration",
                "status": "passed"
            })
            
        except Exception as e:
            self.test_results.append({
                "database": db_name,
                "test": "database_integration",
                "status": "failed",
                "error": str(e)
            })
    
    def generate_report(self):
        """Generate integration test report"""
        passed_tests = len([r for r in self.test_results if r["status"] == "passed"])
        total_tests = len(self.test_results)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
            },
            "details": self.test_results,
            "generated_at": datetime.now()
        }
        
        return report
```

This comprehensive integration setup guide provides NEO with the capability to connect with virtually any external system, from simple APIs to complex enterprise applications, ensuring seamless workflow automation and data synchronization across all platforms.
