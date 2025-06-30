# Chapter 9: PC Control & Automation
**Complete System Control and Intelligent Automation**

---

## 9.1 Overview of PC Control

NEO provides comprehensive control over your computer system, enabling seamless automation of routine tasks and complex operations. From simple commands like shutdown to sophisticated system management, NEO serves as your intelligent system administrator.

### Key Capabilities
- **Power Management**: Shutdown, restart, hibernate, sleep control
- **Process Management**: Start, stop, monitor applications and services
- **System Monitoring**: Real-time performance tracking and alerts
- **Hardware Control**: CPU, GPU, memory, and peripheral management
- **Automated Maintenance**: System optimization and cleanup
- **Security Operations**: Real-time threat monitoring and response

## 9.2 Basic System Commands

### Power Management
```bash
# Shutdown system
neo "shutdown the computer"
neo "shutdown in 10 minutes"
neo "schedule shutdown at 11 PM"

# Restart system
neo "restart the computer"
neo "reboot now"
neo "restart and check for updates"

# Sleep and hibernation
neo "put computer to sleep"
neo "hibernate now"
neo "sleep after 30 minutes of inactivity"
```

### System Information
```bash
# System status
neo "show system information"
neo "what's my CPU usage?"
neo "check memory usage"
neo "display disk space"

# Hardware details
neo "list all connected devices"
neo "show graphics card information"
neo "check network adapters"
```

## 9.3 Advanced System Operations

### Process Management
```bash
# Application control
neo "start Microsoft Word"
neo "close all browser windows"
neo "kill process chrome.exe"
neo "restart the print spooler service"

# Performance monitoring
neo "show me the top 10 CPU consuming processes"
neo "alert me if any process uses more than 80% CPU"
neo "terminate unresponsive applications"
```

### System Maintenance
```bash
# Cleanup operations
neo "clean temporary files"
neo "empty recycle bin"
neo "clear browser cache"
neo "defragment drive C"

# System optimization
neo "optimize startup programs"
neo "update all drivers"
neo "scan for system errors"
neo "optimize system performance"
```

## 9.4 File System Management

### File Operations
```bash
# File management
neo "find all PDF files modified today"
neo "backup Documents folder to external drive"
neo "organize Downloads folder by file type"
neo "delete files older than 30 days in Temp folder"

# Permission management
neo "change file permissions for project folder"
neo "make file read-only"
neo "grant full access to user John"
```

### Directory Operations
```bash
# Directory management
neo "create folder structure for new project"
neo "compress folder to ZIP file"
neo "sync folder with cloud storage"
neo "monitor folder for changes"
```

## 9.5 Network Management

### Network Configuration
```bash
# Network operations
neo "show network configuration"
neo "reset network adapters"
neo "flush DNS cache"
neo "test internet connectivity"

# Wi-Fi management
neo "connect to Wi-Fi network HomeNetwork"
neo "show saved Wi-Fi passwords"
neo "scan for available networks"
neo "forget Wi-Fi network GuestNetwork"
```

### Network Monitoring
```bash
# Traffic monitoring
neo "show network usage by application"
neo "monitor bandwidth usage"
neo "alert if data usage exceeds 10GB"
neo "block internet access for specific app"
```

## 9.6 Security Operations

### System Security
```bash
# Security monitoring
neo "scan system for malware"
neo "check for security updates"
neo "monitor login attempts"
neo "enable firewall with strict rules"

# Access control
neo "lock workstation"
neo "log out current user"
neo "disable USB ports"
neo "enable screen saver with password"
```

### Threat Response
```bash
# Incident response
neo "isolate system from network"
neo "backup critical files immediately"
neo "scan for rootkits"
neo "enable maximum security mode"
```

## 9.7 Hardware Control

### CPU and Memory
```bash
# Performance control
neo "set CPU to power saving mode"
neo "allocate more memory to application X"
neo "limit CPU usage for background processes"
neo "optimize memory usage"
```

### Graphics and Display
```bash
# Display management
neo "change screen resolution to 1920x1080"
neo "extend display to second monitor"
neo "adjust brightness to 70%"
neo "enable night mode"
```

### Peripheral Control
```bash
# Device management
neo "eject USB drive safely"
neo "disable touchpad"
neo "configure printer settings"
neo "update webcam drivers"
```

## 9.8 Automation Scripts

### Creating Automation Workflows
```bash
# Schedule automated tasks
neo "create daily backup routine"
neo "automate system cleanup every Sunday"
neo "schedule virus scan every night"
neo "auto-start development environment at 9 AM"
```

### Custom Automation Examples

#### Morning Startup Routine
```python
# NEO Automation Script
@neo.schedule("weekdays", "8:00 AM")
def morning_routine():
    neo.system.startup_applications([
        "Outlook", "Teams", "Visual Studio Code"
    ])
    neo.network.check_connectivity()
    neo.system.update_check()
    neo.display.set_brightness(80)
    neo.notify("Good morning! Your system is ready.")
```

#### End of Day Cleanup
```python
@neo.schedule("daily", "6:00 PM")
def evening_cleanup():
    neo.files.cleanup_temp()
    neo.system.close_non_essential_apps()
    neo.backup.incremental_backup("Documents")
    neo.security.scan_quick()
    neo.notify("Daily cleanup completed.")
```

## 9.9 System Monitoring and Alerts

### Performance Monitoring
```bash
# Real-time monitoring
neo "monitor CPU temperature"
neo "alert if memory usage exceeds 90%"
neo "track disk I/O performance"
neo "monitor network latency"
```

### Custom Alerts
```bash
# Alert configuration
neo "notify me when download completes"
neo "alert if antivirus definitions are outdated"
neo "warn if system hasn't been restarted in 7 days"
neo "notify when specific application crashes"
```

## 9.10 Emergency Controls

### System Recovery
```bash
# Emergency procedures
neo "enter safe mode on next restart"
neo "create system restore point"
neo "boot from recovery disk"
neo "reset network settings to default"
```

### Data Protection
```bash
# Emergency backup
neo "emergency backup all user data"
neo "create system image backup"
neo "sync critical files to cloud immediately"
neo "encrypt sensitive data"
```

## 9.11 Multi-System Management

### Remote Control
```bash
# Remote operations (with proper authentication)
neo "shutdown bedroom computer"
neo "check status of server in office"
neo "wake up laptop via WOL"
neo "sync files between all devices"
```

### Fleet Management
```bash
# Enterprise features
neo "update all workstations in IT department"
neo "deploy security patch to all systems"
neo "generate fleet status report"
neo "configure company-wide policies"
```

## 9.12 Integration with Smart Home

### IoT Device Control
```bash
# Smart home integration
neo "turn off all lights when shutting down PC"
neo "adjust room temperature when starting intensive tasks"
neo "activate security cameras during suspicious activity"
neo "control smart speakers for notifications"
```

## 9.13 Troubleshooting System Control Issues

### Common Issues and Solutions

#### Permission Denied Errors
- **Issue**: Insufficient privileges for system operations
- **Solution**: Run NEO with administrative privileges
```bash
# Windows
neo elevate --enable-admin-mode

# Linux/macOS
sudo neo --admin-mode
```

#### Service Access Issues
- **Issue**: Cannot control Windows services
- **Solution**: Enable service control permissions
```bash
neo permissions --enable-service-control
```

### Best Practices
1. **Security First**: Always verify commands before execution
2. **Backup Before Changes**: Create restore points for major operations
3. **Monitor Resource Usage**: Track NEO's own resource consumption
4. **Regular Maintenance**: Keep automation scripts updated
5. **Emergency Access**: Maintain manual control methods as backup

---

**Next Chapter**: [File Management](10-file-management.md)

*With NEO's system control capabilities, your computer becomes an extension of your intelligent digital workspace.*
