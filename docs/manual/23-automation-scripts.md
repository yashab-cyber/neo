# Chapter 23: Automation Scripts

## Overview

NEO's automation scripting system allows you to create sophisticated, reusable scripts that can handle complex multi-step tasks. This chapter covers the creation, management, and execution of automation scripts for various scenarios.

## Script Types

### 1. Task Automation Scripts

```python
# Example: System maintenance script
neo_script = {
    "name": "daily_maintenance",
    "description": "Performs daily system maintenance tasks",
    "steps": [
        {"action": "system.disk_cleanup", "params": {"threshold": 85}},
        {"action": "system.update_check", "params": {"auto_install": True}},
        {"action": "security.scan", "params": {"quick_scan": True}},
        {"action": "backup.create", "params": {"type": "incremental"}}
    ],
    "schedule": "daily_9am",
    "notifications": True
}
```

### 2. Workflow Scripts

```yaml
# workflow_data_processing.yaml
name: "Data Processing Workflow"
version: "1.2"
triggers:
  - type: "file_watch"
    path: "/data/incoming"
    pattern: "*.csv"

steps:
  - name: "validate_data"
    action: "data.validate"
    params:
      schema: "/schemas/data_schema.json"
      
  - name: "process_data"
    action: "data.transform"
    params:
      operations: ["clean", "normalize", "aggregate"]
      
  - name: "generate_report"
    action: "report.create"
    params:
      template: "monthly_summary"
      output: "/reports/processed_data.pdf"
```

## Script Creation Interface

### Visual Script Builder

```bash
# Launch the visual script builder
neo script create --visual

# Available components:
# - Triggers: Time, File, System Event, External API
# - Actions: System, File, Network, Application
# - Conditions: If/Then/Else, Loop, Switch
# - Notifications: Email, Slack, Desktop
```

### Code-Based Scripts

```python
from neo.automation import Script, Trigger, Action

# Create a new script
script = Script("backup_and_sync")

# Add triggers
script.add_trigger(Trigger.schedule("0 2 * * *"))  # Daily at 2 AM
script.add_trigger(Trigger.system_event("low_disk_space"))

# Add actions with conditions
with script.if_condition("disk_usage > 80%"):
    script.add_action(Action.cleanup_temp_files())
    script.add_action(Action.compress_logs())

script.add_action(Action.backup_files("/important_data", "/backup"))
script.add_action(Action.sync_cloud("gdrive://backup"))

# Save and activate
script.save()
script.activate()
```

## Advanced Script Features

### 1. Error Handling and Recovery

```python
# Script with comprehensive error handling
def robust_backup_script():
    try:
        # Primary backup location
        backup_result = neo.backup.create(
            source="/data",
            destination="/backup/primary",
            compression=True
        )
        
        if backup_result.success:
            neo.log("Primary backup completed successfully")
        else:
            # Fallback to secondary location
            backup_result = neo.backup.create(
                source="/data",
                destination="/backup/secondary",
                compression=True
            )
            
    except BackupError as e:
        # Emergency backup to cloud
        neo.cloud.emergency_backup("/data", "emergency_backup")
        neo.notify.admin(f"Backup failed: {e}")
        
    except Exception as e:
        neo.log.error(f"Critical backup failure: {e}")
        neo.system.alert("CRITICAL: Backup system failure")
```

### 2. Dynamic Parameter Adjustment

```python
# Self-adjusting script based on system conditions
class AdaptiveMaintenanceScript:
    def __init__(self):
        self.performance_threshold = 0.8
        self.cleanup_intensity = "normal"
    
    def execute(self):
        # Check system performance
        cpu_usage = neo.system.cpu_usage()
        memory_usage = neo.system.memory_usage()
        
        # Adjust cleanup intensity based on usage
        if cpu_usage > 0.9 or memory_usage > 0.9:
            self.cleanup_intensity = "aggressive"
            neo.log("High system usage detected, increasing cleanup intensity")
        elif cpu_usage < 0.3 and memory_usage < 0.3:
            self.cleanup_intensity = "thorough"
            neo.log("Low system usage, performing thorough cleanup")
        
        # Execute cleanup with adjusted parameters
        neo.system.cleanup(intensity=self.cleanup_intensity)
```

## Script Management

### Script Repository

```bash
# List all available scripts
neo script list

# Search scripts by category
neo script search --category "maintenance"
neo script search --tag "backup"

# Get script information
neo script info daily_maintenance

# View script execution history
neo script history daily_maintenance --last 30
```

### Version Control

```bash
# Create script version
neo script version create daily_maintenance --message "Added disk cleanup"

# List versions
neo script version list daily_maintenance

# Rollback to previous version
neo script version rollback daily_maintenance v1.2

# Compare versions
neo script version diff daily_maintenance v1.2 v1.3
```

## Scheduling and Triggers

### Time-Based Scheduling

```python
# Various scheduling options
scripts = [
    {
        "name": "hourly_check",
        "schedule": "0 * * * *",  # Every hour
        "action": "system.health_check"
    },
    {
        "name": "business_hours_only",
        "schedule": "0 9-17 * * 1-5",  # 9 AM to 5 PM, weekdays
        "action": "productivity.report"
    },
    {
        "name": "monthly_report",
        "schedule": "0 0 1 * *",  # First day of month
        "action": "analytics.monthly_summary"
    }
]
```

### Event-Based Triggers

```python
# Event-driven script execution
event_scripts = {
    "file_modified": {
        "trigger": {"type": "file_change", "path": "/config/*.conf"},
        "action": "system.reload_config"
    },
    "high_cpu": {
        "trigger": {"type": "performance", "metric": "cpu", "threshold": 90},
        "action": "performance.investigate_high_cpu"
    },
    "security_alert": {
        "trigger": {"type": "security_event", "severity": "high"},
        "action": "security.immediate_response"
    }
}
```

## Script Library Examples

### 1. Development Environment Setup

```bash
#!/bin/bash
# dev_setup.sh - Automated development environment setup

neo script execute --inline "
# Update system packages
system.update()

# Install development tools
package.install(['git', 'nodejs', 'python3', 'docker', 'vscode'])

# Configure git
git.config('user.name', prompt('Enter your name:'))
git.config('user.email', prompt('Enter your email:'))

# Setup development directories
file.create_dirs([
    '~/workspace/projects',
    '~/workspace/tools',
    '~/workspace/scripts'
])

# Clone common repositories
git.clone('https://github.com/company/common-tools', '~/workspace/tools/')

# Install Python packages
python.pip_install(['requests', 'pandas', 'numpy', 'flask'])

# Start Docker daemon
service.start('docker')

notify('Development environment setup complete!')
"
```

### 2. Security Monitoring Script

```python
# security_monitor.py
import neo
from datetime import datetime, timedelta

class SecurityMonitor:
    def __init__(self):
        self.alerts = []
        self.last_scan = datetime.now() - timedelta(hours=24)
    
    def continuous_monitoring(self):
        while True:
            # Check for failed login attempts
            failed_logins = neo.security.get_failed_logins(
                since=self.last_scan
            )
            
            if len(failed_logins) > 10:
                self.alert(f"High number of failed logins: {len(failed_logins)}")
            
            # Monitor network connections
            suspicious_connections = neo.network.get_suspicious_connections()
            for conn in suspicious_connections:
                self.alert(f"Suspicious connection: {conn.remote_ip}")
            
            # Check file integrity
            modified_files = neo.security.check_file_integrity()
            for file in modified_files:
                if file.critical:
                    self.alert(f"Critical file modified: {file.path}")
            
            # Update scan time
            self.last_scan = datetime.now()
            
            # Wait before next scan
            neo.sleep(300)  # 5 minutes
    
    def alert(self, message):
        neo.log.security(message)
        neo.notify.security_team(message)
        self.alerts.append({
            "timestamp": datetime.now(),
            "message": message
        })
```

### 3. Data Processing Pipeline

```yaml
# data_pipeline.yaml
name: "Customer Data Processing Pipeline"
description: "Processes customer data from multiple sources"

stages:
  - name: "data_ingestion"
    parallel: true
    tasks:
      - source: "database"
        query: "SELECT * FROM customers WHERE updated_at > ?"
        params: ["{{ last_run_time }}"]
      - source: "api"
        endpoint: "https://api.company.com/customers"
        auth: "bearer_token"
      - source: "csv_files"
        path: "/data/incoming/*.csv"
        
  - name: "data_validation"
    depends_on: ["data_ingestion"]
    rules:
      - field: "email"
        validation: "email_format"
      - field: "phone"
        validation: "phone_format"
      - field: "age"
        validation: "range(0, 150)"
        
  - name: "data_enrichment"
    depends_on: ["data_validation"]
    enrichments:
      - type: "geolocation"
        input: "address"
        output: "coordinates"
      - type: "demographic"
        input: "zip_code"
        output: "demographic_data"
        
  - name: "data_output"
    depends_on: ["data_enrichment"]
    outputs:
      - format: "json"
        destination: "/data/processed/customers.json"
      - format: "parquet"
        destination: "/data/warehouse/customers.parquet"
      - database: "analytics_db"
        table: "processed_customers"
```

## Performance Optimization

### Script Performance Monitoring

```python
# Performance monitoring for scripts
from neo.monitoring import ScriptProfiler

@ScriptProfiler.monitor
def optimized_data_processing():
    # Use parallel processing for large datasets
    with neo.parallel.pool(workers=4) as pool:
        results = pool.map(process_chunk, data_chunks)
    
    # Cache frequently accessed data
    @neo.cache.memoize(ttl=3600)
    def expensive_calculation(data):
        return complex_algorithm(data)
    
    # Use efficient data structures
    processed_data = neo.data.DataFrame(results)
    
    return processed_data

# View performance metrics
neo.script.performance daily_maintenance --detailed
```

### Resource Management

```python
# Resource-aware script execution
class ResourceAwareScript:
    def __init__(self):
        self.max_memory = neo.system.available_memory() * 0.8
        self.max_cpu_cores = neo.system.cpu_count() // 2
    
    def execute_with_limits(self):
        # Set resource limits
        neo.resources.set_limits(
            memory=self.max_memory,
            cpu_cores=self.max_cpu_cores
        )
        
        # Monitor resource usage during execution
        with neo.resources.monitor() as monitor:
            # Execute main script logic
            self.main_logic()
            
            # Check if limits were exceeded
            if monitor.memory_exceeded:
                neo.log.warning("Script exceeded memory limit")
            if monitor.cpu_exceeded:
                neo.log.warning("Script exceeded CPU limit")
```

## Best Practices

### 1. Script Design Principles

- **Modularity**: Break complex scripts into smaller, reusable functions
- **Error Handling**: Always include comprehensive error handling
- **Logging**: Log all important actions and decisions
- **Documentation**: Comment your scripts thoroughly
- **Testing**: Test scripts in safe environments first

### 2. Security Considerations

```python
# Secure script practices
class SecureScript:
    def __init__(self):
        # Use encrypted configuration for sensitive data
        self.config = neo.config.encrypted_load("script_config.enc")
        
        # Validate all inputs
        self.input_validator = neo.security.InputValidator()
    
    def execute_secure_action(self, user_input):
        # Sanitize inputs
        clean_input = self.input_validator.sanitize(user_input)
        
        # Use least privilege principle
        with neo.security.limited_context(permissions=["read_files"]):
            result = self.process_data(clean_input)
        
        # Audit all actions
        neo.audit.log_action("script_execution", {
            "script": self.__class__.__name__,
            "input": clean_input,
            "result": "success"
        })
        
        return result
```

### 3. Maintenance and Updates

```bash
# Script maintenance commands
neo script lint my_script.py
neo script test my_script.py --dry-run
neo script optimize my_script.py
neo script update-dependencies my_script.py
```

## Troubleshooting Scripts

### Common Issues and Solutions

1. **Script Fails to Execute**
   ```bash
   # Check script syntax
   neo script validate my_script.py
   
   # Check permissions
   neo script permissions my_script.py
   
   # View detailed error logs
   neo script logs my_script.py --verbose
   ```

2. **Performance Issues**
   ```bash
   # Profile script performance
   neo script profile my_script.py
   
   # Optimize script
   neo script optimize my_script.py --suggestions
   ```

3. **Dependency Problems**
   ```bash
   # Check dependencies
   neo script dependencies my_script.py
   
   # Update dependencies
   neo script update-deps my_script.py
   ```

This automation scripting system provides NEO with powerful capabilities to handle complex, multi-step tasks efficiently and reliably, making it an essential tool for advanced users and system administrators.
