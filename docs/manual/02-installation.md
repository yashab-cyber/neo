# Chapter 2: Installation & Setup
**Setting Up Your NEO Digital Intelligence System**

---

## 2.1 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Processor**: Intel i5 8th gen or AMD Ryzen 5 3600 equivalent
- **Memory**: 8 GB RAM (16 GB recommended)
- **Storage**: 50 GB available space (SSD recommended)
- **Network**: Stable internet connection (10 Mbps minimum)
- **Graphics**: DirectX 11 compatible (for enhanced visual processing)

### Recommended Requirements
- **Processor**: Intel i7 10th gen or AMD Ryzen 7 5800X
- **Memory**: 32 GB RAM
- **Storage**: 100 GB SSD space
- **Network**: High-speed broadband (50+ Mbps)
- **Graphics**: Dedicated GPU with 4GB+ VRAM (NVIDIA GTX 1060/AMD RX 580 or better)

### Professional/Enterprise Requirements
- **Processor**: Intel i9 or AMD Ryzen 9 series
- **Memory**: 64 GB RAM
- **Storage**: 500 GB NVMe SSD
- **Network**: Enterprise-grade connection
- **Graphics**: Professional GPU (NVIDIA RTX 3070/AMD RX 6800 XT or better)

## 2.2 Pre-Installation Checklist

### Security Considerations
- [ ] Administrative privileges on target system
- [ ] Antivirus software temporarily disabled during installation
- [ ] Firewall configured to allow NEO network access
- [ ] System backup completed (recommended)
- [ ] Previous AI assistant software uninstalled

### Network Configuration
- [ ] Internet connectivity verified
- [ ] DNS resolution working properly
- [ ] Port 443 (HTTPS) accessible
- [ ] Corporate firewall configured (if applicable)

## 2.3 Installation Process

### Step 1: Download NEO
```bash
# Download the latest NEO installer
curl -O https://releases.neo-ai.com/latest/neo-installer.exe  # Windows
curl -O https://releases.neo-ai.com/latest/neo-installer.dmg  # macOS
curl -O https://releases.neo-ai.com/latest/neo-installer.deb  # Linux
```

### Step 2: Verify Installation File
```bash
# Verify installer integrity
sha256sum neo-installer.*
# Compare with published checksums at https://releases.neo-ai.com/checksums
```

### Step 3: Run Installation

#### Windows Installation
1. Right-click `neo-installer.exe` and select "Run as administrator"
2. Follow the installation wizard
3. Select installation directory (default: `C:\Program Files\NEO\`)
4. Choose components:
   - [ ] Core NEO Engine (Required)
   - [ ] Cybersecurity Module
   - [ ] Development Tools
   - [ ] Research Analytics
   - [ ] Voice Interface
5. Configure startup options
6. Complete installation

#### macOS Installation
1. Open `neo-installer.dmg`
2. Drag NEO to Applications folder
3. Open Terminal and run:
```bash
sudo /Applications/NEO.app/Contents/MacOS/setup.sh
```
4. Grant necessary permissions in System Preferences

#### Linux Installation
```bash
# Ubuntu/Debian
sudo dpkg -i neo-installer.deb
sudo apt-get install -f  # Install dependencies

# Configure NEO service
sudo systemctl enable neo
sudo systemctl start neo
```

## 2.4 Initial Configuration

### Step 1: Launch NEO
```bash
# Command line interface
neo --setup

# Or GUI interface (if installed)
neo-gui
```

### Step 2: User Profile Setup
1. **Create User Profile**
   - Full name and preferred name
   - Primary language and region
   - Professional domain/interests
   - Security clearance level (if applicable)

2. **Learning Preferences**
   - Learning style assessment
   - Preferred interaction mode (voice, text, hybrid)
   - Complexity level preferences
   - Domain expertise areas

### Step 3: System Integration
```bash
# Grant system permissions
neo permissions --grant-all

# Or selective permissions
neo permissions --grant shell,filesystem,network
```

### Step 4: Security Configuration
```bash
# Set up encryption keys
neo security --generate-keys

# Configure authentication
neo auth --setup-biometric  # If supported
neo auth --setup-token      # Always available
```

## 2.5 Module Configuration

### Core Modules
```bash
# Enable deep learning capabilities
neo module enable deep-learning

# Enable neuro learning
neo module enable neuro-learning

# Enable recursive learning
neo module enable recursive-learning
```

### Specialized Modules
```bash
# Cybersecurity suite
neo module enable cybersecurity
neo module enable penetration-testing

# Development tools
neo module enable code-analysis
neo module enable debugging-tools

# Research capabilities
neo module enable research-analytics
neo module enable data-mining
```

## 2.6 Network & Cloud Setup

### Local Network Configuration
```bash
# Configure network discovery
neo network --scan-local
neo network --configure-access

# Set up secure communications
neo network --enable-encryption
```

### Cloud Integration (Optional)
```bash
# Connect to NEO Cloud Services
neo cloud --authenticate
neo cloud --sync-profile

# Enable cloud learning (enhanced capabilities)
neo cloud --enable-distributed-learning
```

## 2.7 Verification & Testing

### System Health Check
```bash
# Comprehensive system test
neo diagnostic --full

# Quick verification
neo status --all-modules
```

### Basic Functionality Test
```bash
# Test command interface
neo "What is 2+2?"

# Test system control
neo "Show me system information"

# Test learning capability
neo "Remember that I prefer detailed explanations"
```

## 2.8 Post-Installation Security

### Secure Configuration
```bash
# Enable audit logging
neo security --enable-audit-log

# Set up backup encryption
neo backup --configure-encryption

# Enable threat monitoring
neo security --enable-threat-monitor
```

### Network Security
```bash
# Configure firewall rules
neo security --configure-firewall

# Enable intrusion detection
neo security --enable-ids

# Set up secure communications
neo security --setup-ssl-certs
```

## 2.9 Troubleshooting Installation Issues

### Common Issues and Solutions

#### Installation Fails
- **Issue**: Insufficient permissions
- **Solution**: Run installer as administrator/root
- **Command**: `sudo neo-installer` (Linux/macOS)

#### Module Loading Errors
- **Issue**: Missing dependencies
- **Solution**: Install required libraries
```bash
# Windows
neo deps --install-windows

# Linux
sudo neo deps --install-linux

# macOS
neo deps --install-macos
```

#### Network Connection Issues
- **Issue**: Firewall blocking connections
- **Solution**: Configure firewall exceptions
```bash
neo network --configure-firewall
```

### Getting Help
```bash
# Built-in help system
neo help installation

# Diagnostic information for support
neo diagnostic --support-package

# Community forums
neo community --open-forums
```

---

**Next Chapter**: [First Steps](03-first-steps.md)

*Your NEO installation is now complete. Let's begin your journey into intelligent digital assistance.*
