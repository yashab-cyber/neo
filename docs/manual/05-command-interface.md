# Chapter 5: Command Interface
**Mastering NEO's Natural Language and Structured Command System**

---

## 5.1 Introduction to NEO's Command Interface

NEO features a sophisticated command interface that combines natural language processing with structured commands, enabling both casual users and power users to interact efficiently. Unlike traditional command-line interfaces that require exact syntax, NEO understands context, intent, and natural language variations.

## 5.2 Natural Language Commands

### Conversational Interaction
NEO understands natural language commands as if you're talking to a knowledgeable colleague:

```bash
# Natural language examples
neo "Can you help me understand how Docker works?"
neo "I need to find all Python files in my project directory"
neo "What's causing my computer to run slowly today?"
neo "Schedule a system backup for this weekend"
neo "Show me the weather forecast for tomorrow"
```

### Context Awareness
NEO maintains conversation context and can handle follow-up questions:

```bash
# Initial query
neo "Tell me about machine learning algorithms"

# Follow-up queries (NEO remembers context)
neo "Which one is best for image recognition?"
neo "Can you show me a Python example?"
neo "How does it compare to deep learning?"
neo "What libraries should I use?"
```

### Intent Recognition
NEO recognizes various ways to express the same intent:

```bash
# All of these have the same intent:
neo "shutdown the computer"
neo "turn off my PC"
neo "power down the system"
neo "I want to shut down"
neo "please shutdown"
```

## 5.3 Structured Command Syntax

### Basic Command Structure
```bash
neo <command> [options] [arguments]

# Examples:
neo status                          # Simple command
neo scan --security                 # Command with option
neo backup /home/user/documents     # Command with argument
neo analyze --type performance      # Command with named parameter
```

### Command Categories
```bash
# System commands
neo system <action>
neo power <action>
neo process <action>

# Security commands
neo security <action>
neo scan <target>
neo monitor <service>

# Development commands
neo dev <action>
neo code <action>
neo test <action>

# Learning commands
neo learn <action>
neo train <model>
neo adapt <context>
```

## 5.4 Command Options and Flags

### Global Options
```bash
# Available for all commands
--verbose, -v           # Detailed output
--quiet, -q            # Minimal output
--help, -h             # Show help
--config <file>        # Use specific config
--output <format>      # Output format (json, xml, table)
--timeout <seconds>    # Command timeout
--dry-run              # Preview without executing
--force                # Force execution
--interactive, -i      # Interactive mode
--background, -b       # Run in background
```

### Safety Options
```bash
# Safety and confirmation
--preview              # Show what will be done
--confirm              # Require confirmation
--safe-mode            # Use maximum safety
--no-backup            # Skip automatic backup
--rollback-point       # Create rollback point
```

### Output Control
```bash
# Output formatting
--format json          # JSON output
--format xml           # XML output
--format table         # Tabular output
--format csv           # CSV output
--no-color             # Disable color output
--timestamps           # Include timestamps
--log-file <path>      # Log to file
```

## 5.5 Advanced Command Features

### Command Chaining
```bash
# Chain multiple commands
neo "scan system for vulnerabilities && fix critical issues && generate report"

# Conditional execution
neo "backup documents || alert me if backup fails"

# Parallel execution
neo "update system & scan for malware & check disk space"
```

### Variables and Substitution
```bash
# Define variables
neo set PROJECT_DIR="/home/user/myproject"
neo set BACKUP_LOCATION="/mnt/backup"

# Use variables in commands
neo backup $PROJECT_DIR to $BACKUP_LOCATION
neo analyze security in $PROJECT_DIR
```

### Command Templates
```bash
# Create reusable command templates
neo template create "daily-maintenance" "
  check system health &&
  update security definitions &&
  clean temporary files &&
  backup important documents
"

# Execute template
neo template run "daily-maintenance"
```

## 5.6 Interactive Command Mode

### Starting Interactive Mode
```bash
# Enter interactive mode
neo --interactive

# Or simply
neo -i
```

### Interactive Features
```
NEO> help
Available commands: system, security, dev, learn, analyze...

NEO> system status
✅ System: Healthy
✅ CPU: 15% usage
✅ Memory: 8.2GB/32GB used
✅ Storage: 45% used

NEO> ?security scan
Usage: security scan [target] [options]
Options:
  --type <vulnerability|malware|configuration>
  --depth <quick|standard|deep>
  --report <format>

NEO> security scan --type vulnerability
Starting vulnerability scan...
[████████████████████████████████] 100% Complete
Found 3 low-priority issues. Run 'security scan --report' for details.
```

### Tab Completion
```bash
# Tab completion for commands
NEO> sec<TAB>
security

NEO> security sc<TAB>
scan

NEO> security scan --<TAB>
--type    --depth   --report   --target   --schedule
```

## 5.7 Voice Command Integration

### Enabling Voice Commands
```bash
# Configure voice interface
neo voice --setup
neo voice --calibrate-microphone
neo voice --train-recognition
```

### Voice Command Syntax
```bash
# Wake word activation
"NEO, what time is it?"
"NEO, scan the system for threats"
"NEO, shutdown the computer in 10 minutes"

# Continuous conversation mode
neo voice --continuous
# Then speak naturally without wake word
```

### Voice Command Options
```bash
# Voice-specific settings
neo voice --set-language english-us
neo voice --set-accent american
neo voice --set-sensitivity medium
neo voice --enable-offline-mode
```

## 5.8 Command History and Recall

### History Management
```bash
# View command history
neo history

# Search command history
neo history --search "backup"
neo history --grep "security scan"

# Execute previous command
neo !!

# Execute command by number
neo !42

# Execute last command matching pattern
neo !backup
```

### History with Context
```bash
# Show history with context
NEO> history --with-context

1. [2025-06-29 09:15] scan system --security
   Context: Morning routine security check
   Result: 2 low-priority issues found

2. [2025-06-29 09:20] fix issues --priority low
   Context: Following up on scan results
   Result: Issues resolved successfully

3. [2025-06-29 09:25] backup documents
   Context: Weekly backup routine
   Result: 1.2GB backed up successfully
```

## 5.9 Command Aliasing and Shortcuts

### Creating Aliases
```bash
# Create command aliases
neo alias create "ss" "system status"
neo alias create "backup-docs" "backup /home/user/documents"
neo alias create "morning-check" "
  system status &&
  security scan --quick &&
  check updates
"

# Use aliases
neo ss
neo backup-docs
neo morning-check
```

### Smart Shortcuts
```bash
# NEO learns your patterns and suggests shortcuts
neo suggest-shortcuts

# Example suggestions:
# "You frequently run 'security scan --quick'. Create alias 'qscan'? (y/n)"
# "You often backup documents. Create template 'backup-routine'? (y/n)"
```

## 5.10 Error Handling and Recovery

### Error Messages
```bash
NEO> invalid-command
❌ Command 'invalid-command' not recognized.
💡 Did you mean: 'scan', 'analyze', or 'status'?
📚 Use 'help' to see all available commands.

NEO> scan --invalid-option
❌ Option '--invalid-option' not recognized for command 'scan'.
💡 Available options: --type, --depth, --report, --target
📚 Use 'help scan' for detailed usage information.
```

### Auto-Correction
```bash
# NEO can auto-correct common mistakes
NEO> sytem status
💡 Auto-correcting 'sytem' to 'system'
✅ System: Healthy...

# Or ask for confirmation
NEO> scna security
❓ Did you mean 'scan security'? (y/n)
```

### Recovery Suggestions
```bash
# When commands fail, NEO suggests alternatives
NEO> backup /nonexistent/path
❌ Path '/nonexistent/path' does not exist.
💡 Suggestions:
   - Check if path is correct
   - Use 'find' to locate the directory
   - Create the directory with 'mkdir'
   - Browse available paths with 'ls'
```

## 5.11 Command Documentation

### Inline Help
```bash
# Get help for any command
neo help
neo help security
neo help security scan
neo help backup --examples

# Quick help with ?
neo ?scan
neo ?backup
```

### Command Examples
```bash
# Show examples for commands
neo examples backup
neo examples security scan
neo examples automation

# Output:
# Examples for 'backup':
# 1. Basic backup:     neo backup /home/user/documents
# 2. Scheduled backup: neo backup /data --schedule daily
# 3. Encrypted backup: neo backup /secrets --encrypt
# 4. Cloud backup:     neo backup /projects --to-cloud
```

### Man-Page Style Help
```bash
# Detailed help in man-page format
neo man backup
neo man security
neo man system

# Shows:
# - Command description
# - Detailed options
# - Examples
# - Related commands
# - See also references
```

## 5.12 Command Customization

### Personal Command Preferences
```bash
# Set default options for commands
neo preferences set scan.default-depth standard
neo preferences set backup.default-location /mnt/backup
neo preferences set output.default-format table

# View current preferences
neo preferences show
```

### Custom Command Development
```bash
# Create custom commands using scripts
neo custom create "project-setup" --script setup-project.sh
neo custom create "deploy-app" --script deploy.py
neo custom create "team-report" --command "
  analyze performance --team &&
  generate report --format executive &&
  email report to team-leads
"
```

## 5.13 Security Considerations

### Command Authorization
```bash
# Commands require appropriate permissions
NEO> shutdown system
🔐 Command 'shutdown' requires administrator privileges.
💡 Run 'neo elevate' to gain administrative access.

NEO> delete --recursive /system
⚠️  Dangerous command detected. This will delete system files.
🔐 Administrative confirmation required.
📝 Type 'I understand the risks' to proceed:
```

### Audit Logging
```bash
# All commands are logged for security
neo audit show
neo audit search "delete"
neo audit export --format csv

# Audit log includes:
# - Timestamp
# - User context
# - Command executed
# - Result/status
# - Security level
```

## 5.14 Performance Optimization

### Command Caching
```bash
# Enable command result caching
neo cache --enable
neo cache --set-ttl 3600  # 1 hour cache

# Commands that benefit from caching:
# - System information queries
# - File system scans
# - Network topology discovery
# - Configuration analysis
```

### Parallel Execution
```bash
# Enable parallel command execution
neo parallel --enable
neo parallel --max-workers 4

# Commands automatically parallelized:
# - Multiple file operations
# - Network scans
# - Security checks
# - System monitoring
```

---

**Next Chapter**: [Voice Commands](06-voice-commands.md)

*The command interface is your gateway to NEO's capabilities. Master these patterns and you'll be communicating with NEO as naturally as with a colleague.*
