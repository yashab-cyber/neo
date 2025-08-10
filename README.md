# 🔹 NEO: Neural Executive Operator

**The Next-Generation AI Assistant for Complete Digital Transformation**

NEO (Neural Executive Operator) is an advanced artificial intelligence assistant designed to revolutionize how you interact with technology. Combining deep learning, neuro learning, recursive learning, and smart thinking capabilities, NEO serves as your ultimate digital agent for handling complex tasks, automations, and intelligent decision-making.

## 🚀 Key Features

- **🧠 Advanced Learning Systems**: Deep learning, neuro learning, and recursive learning capabilities
- **🎯 Smart Problem Solving**: Tackles complex problems across mathematics, science, and any subject domain
- **🖥️ Complete PC Control**: Full system automation including shutdown, startup, and complex operations
- **🔒 Cybersecurity Expert**: Advanced penetration testing, security analysis, and threat detection
- **💻 Coding Assistant**: Complete development support with debugging, optimization, and best practices
- **🔬 Research & Development**: Intelligent research capabilities with optimized output generation
- **⚡ Task Automation**: Efficient command execution and decision-making processes
- **🤖 Digital Agent**: Comprehensive action handling and intelligent conversations

## 📚 Documentation Structure

- **[User Manual](docs/manual/)** - Complete 50+ page manual with detailed instructions
- **[Technical Documentation](docs/technical/)** - Architecture, APIs, and implementation details
- **[Research Papers](docs/research/)** - Academic research and development documentation
- **[User Guide](docs/user-guide/)** - Quick start and feature guides

## 🎯 Mission Statement

NEO aims to be your complete digital companion, capable of understanding complex requirements, making intelligent decisions, and executing tasks with precision and efficiency. From simple PC operations to advanced cybersecurity tasks, NEO adapts to your needs and grows with your requirements.

---

*Experience the future of AI assistance with NEO - where intelligence meets action.*

## ⚡ Quick Start

Install (Linux / macOS):

```bash
./install.sh       # dev install (editable, dev deps)
./install.sh --prod
```

Install (Windows PowerShell):

```powershell
pwsh -ExecutionPolicy Bypass -File install.ps1
```

Run the API (three equivalent options):

```bash
neo serve --reload                # via CLI (recommended for dev)
python -m neo serve               # module entry point
python run.py --host 0.0.0.0 --port 8000  # convenience launcher
```

Environment overrides supported by `run.py`:

```
NEO_HOST / HOST, NEO_PORT / PORT, NEO_RELOAD, NEO_WORKERS, NEO_LOG_LEVEL, NEO_ROOT_PATH
```

Chat test:

```bash
neo chat "hello neo"
```

Memory example:

```bash
neo memory store "remember this"
neo memory status
```

Knowledge graph seed + status:

```bash
neo knowledge node-create --type Person --props '{"name":"Alice"}'
neo knowledge status
```

For more commands run:

```bash
neo -h
```