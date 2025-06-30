# User Guide: Voice Command Setup

## Overview

NEO's voice command system enables hands-free interaction using natural speech. This guide covers setting up voice recognition, configuring voice commands, and optimizing performance for your specific environment and needs.

## Prerequisites

### Hardware Requirements

```yaml
minimum_requirements:
  microphone: "Built-in or USB microphone"
  audio_quality: "16-bit, 16kHz minimum"
  cpu: "Dual-core processor"
  ram: "4GB available"
  storage: "500MB for voice models"

recommended_requirements:
  microphone: "Dedicated USB microphone or headset"
  audio_quality: "24-bit, 48kHz"
  cpu: "Quad-core processor or better"
  ram: "8GB available"
  storage: "2GB for advanced voice models"
  
optimal_requirements:
  microphone: "Professional USB microphone with noise cancellation"
  audio_quality: "32-bit, 96kHz"
  cpu: "8-core processor"
  ram: "16GB available"
  storage: "4GB for all voice models and languages"
```

### Software Dependencies

```bash
# Install required audio libraries (Linux)
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio alsa-utils

# Install required audio libraries (macOS)
brew install portaudio
pip install pyaudio

# Install required audio libraries (Windows)
# PyAudio is typically included with NEO installer
pip install pyaudio
```

## Initial Voice Setup

### 1. Voice Recognition Configuration

```bash
# Start voice setup wizard
neo voice setup

# Test microphone
neo voice test-microphone

# Configure audio input device
neo voice configure-device --list-devices
neo voice configure-device --select-device "USB Microphone"

# Set audio quality
neo voice set-quality --sample-rate 48000 --bit-depth 24
```

### 2. Voice Model Installation

```bash
# Install default English voice model
neo voice install-model en-US

# Install additional language models
neo voice install-model es-ES  # Spanish
neo voice install-model fr-FR  # French
neo voice install-model de-DE  # German
neo voice install-model ja-JP  # Japanese

# Install specialized models
neo voice install-model en-US-technical  # Technical vocabulary
neo voice install-model en-US-medical    # Medical terminology
neo voice install-model en-US-legal      # Legal terminology

# List available models
neo voice list-models --available

# Check installed models
neo voice list-models --installed
```

### 3. Voice Profile Creation

```bash
# Create personal voice profile
neo voice create-profile --name "primary"

# Voice training session (recommended)
neo voice train-profile --profile "primary" --sessions 5

# Import voice profile from backup
neo voice import-profile --file "voice_profile_backup.json"
```

## Voice Training Process

### Training Session Setup

```python
# Interactive voice training
class VoiceTrainingSession:
    def __init__(self):
        self.training_phrases = [
            "Hello NEO, what's the system status?",
            "Show me the file management interface",
            "Execute a security scan of the network",
            "Generate a performance report for the last week",
            "Create a backup of the important documents folder",
            "What's the current CPU and memory usage?",
            "Schedule a task to run tomorrow morning",
            "Open the artificial intelligence settings",
            "Analyze the recent log files for errors",
            "Help me optimize the system performance"
        ]
    
    def start_training(self, profile_name):
        """Start interactive voice training session"""
        print(f"Starting voice training for profile: {profile_name}")
        print("Please repeat the following phrases clearly:")
        
        for i, phrase in enumerate(self.training_phrases, 1):
            print(f"\nPhrase {i}/{len(self.training_phrases)}:")
            print(f"Say: '{phrase}'")
            
            # Record user speech
            audio_data = self.record_phrase(timeout=10)
            
            # Process and validate
            recognized_text = self.process_audio(audio_data)
            confidence = self.calculate_confidence(phrase, recognized_text)
            
            print(f"Recognized: '{recognized_text}' (Confidence: {confidence:.2f})")
            
            if confidence < 0.8:
                print("Low confidence detected. Please try again.")
                # Retry logic here
                
        self.save_training_data(profile_name)
        print("Voice training completed successfully!")

# Usage
trainer = VoiceTrainingSession()
trainer.start_training("primary")
```

### Advanced Training Options

```bash
# Adaptive training (learns from your usage)
neo voice enable-adaptive-training --profile "primary"

# Environment-specific training
neo voice train-environment --environment "office" --noise-level "medium"
neo voice train-environment --environment "home" --noise-level "low"
neo voice train-environment --environment "mobile" --noise-level "high"

# Accent adaptation
neo voice adapt-accent --profile "primary" --accent "british"
neo voice adapt-accent --profile "primary" --accent "australian"

# Technical vocabulary training
neo voice train-vocabulary --domain "programming"
neo voice train-vocabulary --domain "system-administration"
neo voice train-vocabulary --domain "cybersecurity"
```

## Voice Command Configuration

### Basic Voice Commands

```yaml
# voice_commands.yaml
basic_commands:
  activation:
    wake_words: ["Hey NEO", "NEO", "Computer"]
    sensitivity: 0.7
    timeout: 5  # seconds
    
  system_control:
    - pattern: "show system status"
      command: "system.status"
      confirmation: false
      
    - pattern: "what is the [cpu|memory|disk] usage"
      command: "system.get_usage"
      parameters:
        metric: "{captured_group_1}"
      confirmation: false
      
    - pattern: "shutdown system"
      command: "system.shutdown"
      confirmation: true
      confirmation_phrase: "Are you sure you want to shutdown the system?"
      
    - pattern: "restart [system|computer]"
      command: "system.restart"
      confirmation: true
      
  file_operations:
    - pattern: "create [file|folder] {name}"
      command: "files.create"
      parameters:
        type: "{captured_group_1}"
        name: "{name}"
        
    - pattern: "find files containing {search_term}"
      command: "files.search"
      parameters:
        query: "{search_term}"
        type: "content"
        
    - pattern: "backup {folder_name} to {destination}"
      command: "files.backup"
      parameters:
        source: "{folder_name}"
        destination: "{destination}"
```

### Advanced Voice Command Patterns

```python
class VoiceCommandProcessor:
    def __init__(self):
        self.command_patterns = {}
        self.context_stack = []
        self.variable_slots = {}
        
    def register_dynamic_command(self, pattern, handler, context=None):
        """Register dynamic voice command with context awareness"""
        compiled_pattern = self.compile_pattern(pattern)
        
        self.command_patterns[pattern] = {
            'compiled': compiled_pattern,
            'handler': handler,
            'context': context,
            'variables': self.extract_variables(pattern)
        }
    
    def process_voice_input(self, audio_input):
        """Process voice input with context awareness"""
        # Convert audio to text
        text = self.speech_to_text(audio_input)
        
        # Apply current context
        contextualized_text = self.apply_context(text)
        
        # Match against patterns
        matched_command = self.match_command_pattern(contextualized_text)
        
        if matched_command:
            return self.execute_command(matched_command, contextualized_text)
        else:
            return self.handle_unrecognized_command(text)
    
    def apply_context(self, text):
        """Apply conversational context to improve recognition"""
        if self.context_stack:
            current_context = self.context_stack[-1]
            
            # Resolve pronouns and references
            if "it" in text.lower():
                text = text.replace("it", current_context.get('last_entity', 'it'))
            
            if "there" in text.lower():
                text = text.replace("there", current_context.get('last_location', 'there'))
                
        return text

# Example usage
processor = VoiceCommandProcessor()

# Register contextual commands
processor.register_dynamic_command(
    "create a backup of {folder}",
    handler=backup_handler,
    context="file_operations"
)

processor.register_dynamic_command(
    "and compress it with {compression_type}",
    handler=compression_handler,
    context="file_operations"
)

# Multi-turn conversation example:
# User: "Create a backup of my documents folder"
# NEO: "Backup created. Where would you like to store it?"
# User: "Put it in the cloud storage"
# NEO: "Backup uploaded to cloud storage successfully."
```

### Custom Command Creation

```bash
# Create custom voice command
neo voice create-command \
  --pattern "start development environment for {project}" \
  --command "dev.start_environment" \
  --parameters "project={project}" \
  --confirmation false

# Create command with multiple variations
neo voice create-command \
  --pattern "launch|start|open development environment" \
  --command "dev.start_environment" \
  --parameters "project=default"

# Create command with conditional logic
neo voice create-command \
  --pattern "deploy {application} to {environment}" \
  --command "deploy.application" \
  --conditions "environment in ['staging', 'production']" \
  --confirmation "environment == 'production'"
```

## Voice Interface Customization

### Personal Voice Settings

```python
# voice_settings.py
class PersonalVoiceSettings:
    def __init__(self, user_profile):
        self.user_profile = user_profile
        self.preferences = self.load_preferences()
        
    def configure_voice_personality(self):
        """Configure NEO's voice personality and responses"""
        personality_config = {
            'response_style': 'professional',  # casual, professional, friendly
            'verbosity': 'medium',             # brief, medium, detailed
            'confirmation_level': 'smart',     # always, never, smart
            'error_handling': 'helpful',       # brief, helpful, detailed
            'proactive_suggestions': True,
            'learning_from_corrections': True
        }
        
        return personality_config
    
    def set_response_preferences(self):
        """Set how NEO responds to voice commands"""
        response_prefs = {
            'acknowledge_commands': True,
            'provide_status_updates': True,
            'suggest_alternatives': True,
            'ask_clarification': True,
            'remember_preferences': True
        }
        
        # Voice feedback settings
        voice_feedback = {
            'success_sounds': True,
            'error_sounds': True,
            'thinking_sounds': False,
            'completion_chimes': True
        }
        
        return {
            'responses': response_prefs,
            'audio_feedback': voice_feedback
        }

# Apply settings
settings = PersonalVoiceSettings("user_profile")
neo.voice.configure(settings.configure_voice_personality())
neo.voice.set_feedback(settings.set_response_preferences())
```

### Environmental Adaptation

```bash
# Configure for different environments
neo voice create-environment "office" \
  --noise-cancellation high \
  --microphone-sensitivity 0.8 \
  --wake-word-threshold 0.6

neo voice create-environment "home" \
  --noise-cancellation medium \
  --microphone-sensitivity 0.7 \
  --wake-word-threshold 0.5

neo voice create-environment "mobile" \
  --noise-cancellation maximum \
  --microphone-sensitivity 0.9 \
  --wake-word-threshold 0.8

# Automatic environment detection
neo voice enable-auto-environment \
  --use-location true \
  --use-audio-profile true \
  --use-time-of-day true
```

## Troubleshooting Voice Issues

### Common Problems and Solutions

```python
class VoiceTroubleshooter:
    def __init__(self):
        self.diagnostic_tests = [
            'microphone_test',
            'audio_driver_test',
            'noise_level_test',
            'recognition_accuracy_test',
            'latency_test'
        ]
    
    def diagnose_voice_issues(self):
        """Run comprehensive voice system diagnostics"""
        results = {}
        
        for test in self.diagnostic_tests:
            try:
                result = getattr(self, test)()
                results[test] = result
            except Exception as e:
                results[test] = {'status': 'failed', 'error': str(e)}
        
        return self.generate_diagnostic_report(results)
    
    def microphone_test(self):
        """Test microphone functionality"""
        # Record test audio
        test_audio = self.record_test_audio(duration=3)
        
        # Analyze audio quality
        audio_metrics = {
            'volume_level': self.calculate_volume_level(test_audio),
            'noise_floor': self.calculate_noise_floor(test_audio),
            'signal_to_noise_ratio': self.calculate_snr(test_audio),
            'frequency_response': self.analyze_frequency_response(test_audio)
        }
        
        # Determine if microphone is working properly
        if audio_metrics['volume_level'] < 0.1:
            return {'status': 'failed', 'issue': 'microphone_not_detected'}
        elif audio_metrics['signal_to_noise_ratio'] < 10:
            return {'status': 'warning', 'issue': 'high_noise_level'}
        else:
            return {'status': 'passed', 'metrics': audio_metrics}
    
    def recognition_accuracy_test(self):
        """Test speech recognition accuracy"""
        test_phrases = [
            "Hello NEO",
            "What is the system status?",
            "Create a new file",
            "Show me the current time",
            "Execute system diagnostics"
        ]
        
        accuracy_scores = []
        for phrase in test_phrases:
            print(f"Please say: '{phrase}'")
            recognized = self.get_recognized_text()
            accuracy = self.calculate_phrase_accuracy(phrase, recognized)
            accuracy_scores.append(accuracy)
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        
        return {
            'status': 'passed' if avg_accuracy > 0.8 else 'warning',
            'average_accuracy': avg_accuracy,
            'individual_scores': accuracy_scores
        }

# Usage
troubleshooter = VoiceTroubleshooter()
diagnostic_results = troubleshooter.diagnose_voice_issues()
```

### Performance Optimization

```bash
# Optimize voice recognition performance
neo voice optimize --profile "primary"

# Reduce latency
neo voice set-processing-mode "realtime"
neo voice enable-hardware-acceleration

# Improve accuracy
neo voice retrain-problematic-commands
neo voice update-vocabulary --from-usage-logs

# Audio preprocessing optimization
neo voice configure-preprocessing \
  --noise-reduction "adaptive" \
  --echo-cancellation true \
  --automatic-gain-control true

# Model optimization
neo voice optimize-model \
  --target-accuracy 0.95 \
  --max-latency 200ms \
  --memory-limit 2GB
```

### Advanced Troubleshooting

```python
# Advanced voice debugging
class AdvancedVoiceDebugger:
    def debug_recognition_pipeline(self, audio_input):
        """Debug the entire voice recognition pipeline"""
        debug_info = {}
        
        # Stage 1: Audio preprocessing
        preprocessed_audio = self.debug_preprocessing(audio_input)
        debug_info['preprocessing'] = preprocessed_audio['debug_info']
        
        # Stage 2: Feature extraction
        features = self.debug_feature_extraction(preprocessed_audio['audio'])
        debug_info['feature_extraction'] = features['debug_info']
        
        # Stage 3: Speech recognition
        recognition_result = self.debug_speech_recognition(features['features'])
        debug_info['speech_recognition'] = recognition_result['debug_info']
        
        # Stage 4: Intent classification
        intent_result = self.debug_intent_classification(recognition_result['text'])
        debug_info['intent_classification'] = intent_result['debug_info']
        
        # Stage 5: Command execution
        execution_result = self.debug_command_execution(intent_result['intent'])
        debug_info['command_execution'] = execution_result['debug_info']
        
        return debug_info
    
    def generate_debug_report(self, debug_info):
        """Generate comprehensive debug report"""
        report = {
            'pipeline_stages': [],
            'bottlenecks': [],
            'recommendations': []
        }
        
        for stage, info in debug_info.items():
            stage_report = {
                'stage': stage,
                'processing_time': info.get('processing_time', 0),
                'accuracy': info.get('accuracy', 0),
                'confidence': info.get('confidence', 0),
                'errors': info.get('errors', [])
            }
            report['pipeline_stages'].append(stage_report)
            
            # Identify bottlenecks
            if info.get('processing_time', 0) > 100:  # ms
                report['bottlenecks'].append(f"High latency in {stage}")
            
            if info.get('accuracy', 1.0) < 0.8:
                report['bottlenecks'].append(f"Low accuracy in {stage}")
        
        return report

# Usage for debugging
debugger = AdvancedVoiceDebugger()
debug_info = debugger.debug_recognition_pipeline(audio_input)
report = debugger.generate_debug_report(debug_info)
```

This comprehensive voice setup guide ensures users can configure and optimize NEO's voice interface for their specific needs, environment, and preferences, providing a seamless hands-free experience.
