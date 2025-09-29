# Production Deployment Guide

This guide covers deploying GlassAlpha in production environments, with special attention to security, compliance, and regulatory requirements.

## Deployment Overview

GlassAlpha is designed for **secure, on-premise deployment** in regulated industries. Key characteristics:

- **Local processing only** - No external API calls or cloud dependencies
- **File-based storage** - No databases required
- **Deterministic operations** - Reproducible results for compliance
- **Audit trail generation** - Complete lineage tracking
- **Enterprise-ready** - Role-based access and advanced features available

## Pre-Deployment Planning

### Environment Requirements

#### Production Server Specifications:
- **CPU**: 4+ cores (8+ recommended for large datasets)
- **Memory**: 16GB+ RAM (32GB+ for datasets >100K rows)
- **Storage**: 100GB+ SSD (audit reports, manifests, logs)
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+), macOS 10.15+, Windows Server 2019+
- **Python**: 3.11+ with virtual environment support
- **Network**: Air-gapped capability (no internet required for core operations)

#### Security Requirements:
- Hardened operating system with latest security updates
- Restricted user accounts with minimal privileges
- Encrypted file systems for data at rest
- Secure network configuration (firewall rules, VPN access)
- Audit logging enabled at OS level

### Regulatory Considerations

#### Compliance Framework Alignment:
- **SOC 2 Type II** - Access controls, availability, confidentiality
- **ISO 27001** - Information security management
- **GDPR** - Data protection and privacy requirements
- **HIPAA** - Healthcare data protection (if applicable)
- **SOX** - Financial reporting controls (if applicable)

#### Audit Requirements:
- Complete audit trails of all system activities
- Data lineage and transformation tracking
- User access and activity logging
- Configuration change management
- Incident response and reporting procedures

## Installation and Setup

### 1. Environment Preparation

**Create dedicated service account:**
```bash
# Linux/macOS
sudo useradd -m -s /bin/bash glassalpha
sudo mkdir -p /opt/glassalpha/{app,data,logs,config}
sudo chown -R glassalpha:glassalpha /opt/glassalpha
```

**Set up directory structure:**
```
/opt/glassalpha/
├── app/                    # Application code
├── data/                   # Input datasets
├── config/                 # Configuration files
├── output/                 # Generated reports and manifests
├── logs/                   # Application and audit logs
├── backups/                # Configuration and data backups
└── temp/                   # Temporary processing files
```

### 2. Application Installation

**Install as service account:**
```bash
sudo su - glassalpha

# Create Python virtual environment
python3.11 -m venv /opt/glassalpha/app/venv
source /opt/glassalpha/app/venv/bin/activate

# Install GlassAlpha
cd /opt/glassalpha/app
git clone https://github.com/GlassAlpha/glassalpha.git
cd glassalpha/packages
pip install --upgrade pip
pip install -e .

# Verify installation
glassalpha --version
glassalpha list
```

**Set environment variables:**
```bash
# /opt/glassalpha/.env
export GLASSALPHA_CONFIG_DIR="/opt/glassalpha/config"
export GLASSALPHA_DATA_DIR="/opt/glassalpha/data"
export GLASSALPHA_OUTPUT_DIR="/opt/glassalpha/output"
export GLASSALPHA_LOG_LEVEL="INFO"
export GLASSALPHA_LOG_FILE="/opt/glassalpha/logs/glassalpha.log"

# Enterprise license (if applicable)
# export GLASSALPHA_LICENSE_KEY="your-enterprise-key"

# Load environment
source /opt/glassalpha/.env
```

### 3. System Integration

**Create systemd service (Linux):**
```ini
# /etc/systemd/system/glassalpha.service
[Unit]
Description=GlassAlpha ML Audit Service
After=network.target

[Service]
Type=notify
User=glassalpha
Group=glassalpha
WorkingDirectory=/opt/glassalpha/app
Environment=PATH=/opt/glassalpha/app/venv/bin
EnvironmentFile=/opt/glassalpha/.env
ExecStart=/opt/glassalpha/app/venv/bin/glassalpha serve
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/glassalpha

[Install]
WantedBy=multi-user.target
```

**Enable and start service:**
```bash
sudo systemctl enable glassalpha
sudo systemctl start glassalpha
sudo systemctl status glassalpha
```

## Configuration Management

### Production Configuration Structure

**Environment-specific configs:**
```
/opt/glassalpha/config/
├── environments/
│   ├── production.yaml     # Production-specific settings
│   ├── staging.yaml        # Staging environment
│   └── development.yaml    # Development/testing
├── profiles/
│   ├── financial.yaml      # Financial services compliance
│   ├── healthcare.yaml     # Healthcare compliance
│   └── standard.yaml       # General compliance
├── models/
│   ├── prod_model_v1.pkl  # Production model artifacts
│   └── model_configs/     # Model-specific configurations
└── templates/
    ├── standard_audit.yaml # Standard audit configuration
    └── regulatory_audit.yaml # Enhanced regulatory audit
```

**Production configuration template:**
```yaml
# /opt/glassalpha/config/environments/production.yaml
audit_profile: financial_compliance

# Strict mode enforces regulatory requirements
strict_mode: true

# Reproducibility settings
reproducibility:
  random_seed: ${AUDIT_SEED:-42}
  track_git_sha: true
  track_environment: true
  require_data_hash: true

# Data configuration
data:
  base_path: /opt/glassalpha/data
  schema_validation: strict
  pii_detection: enabled
  backup_original: true

# Model configuration
model:
  type: xgboost
  path: /opt/glassalpha/config/models/prod_model_v1.pkl
  validation_required: true

# Security settings
security:
  audit_logging: enabled
  access_logging: enabled
  data_encryption: true
  output_sanitization: true

# Performance settings
performance:
  max_memory_gb: 8
  max_processing_time: 300
  parallel_processing: true
  n_jobs: 4

# Output configuration
output:
  base_path: /opt/glassalpha/output
  retention_days: 2555  # 7 years for regulatory compliance
  backup_enabled: true
  compression: true

# Enterprise features (if licensed)
enterprise:
  enabled: ${GLASSALPHA_ENTERPRISE:-false}
  monitoring: true
  advanced_templates: true
  rbac_enabled: true
```

### Configuration Validation

**Pre-deployment validation:**
```bash
# Validate configuration
glassalpha validate --config config/environments/production.yaml --strict

# Test with sample data
glassalpha audit \
  --config config/environments/production.yaml \
  --config config/templates/standard_audit.yaml \
  --output output/validation_test.pdf \
  --dry-run
```

**Configuration testing pipeline:**
```bash
#!/bin/bash
# config-validation.sh

set -e

echo "Validating production configurations..."

# Test all environment configs
for config in config/environments/*.yaml; do
    echo "Validating $config..."
    glassalpha validate --config "$config" --strict
done

# Test audit profiles
for profile in config/profiles/*.yaml; do
    echo "Testing profile $profile..."
    glassalpha audit --config "$profile" --output /tmp/test.pdf --dry-run
done

echo "All configurations validated successfully!"
```

## Security Considerations

### Access Control

**File system permissions:**
```bash
# Application files - read-only
chmod -R 755 /opt/glassalpha/app
chown -R root:glassalpha /opt/glassalpha/app

# Configuration - restricted access
chmod -R 750 /opt/glassalpha/config
chown -R glassalpha:glassalpha /opt/glassalpha/config

# Data directories - service account only
chmod -R 700 /opt/glassalpha/data
chmod -R 700 /opt/glassalpha/output
chown -R glassalpha:glassalpha /opt/glassalpha/{data,output}

# Logs - append only
chmod -R 640 /opt/glassalpha/logs
chown -R glassalpha:adm /opt/glassalpha/logs
```

**Network security:**
```bash
# Firewall configuration (example for Linux)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow from 10.0.0.0/8 to any port 8080  # Internal network only
sudo ufw enable
```

### Data Protection

**Encryption at rest:**
```bash
# Encrypt sensitive directories using LUKS (Linux)
sudo cryptsetup luksFormat /dev/sdb1
sudo cryptsetup luksOpen /dev/sdb1 glassalpha-data
sudo mkfs.ext4 /dev/mapper/glassalpha-data
sudo mount /dev/mapper/glassalpha-data /opt/glassalpha/data

# Add to /etc/fstab for automatic mounting
echo "/dev/mapper/glassalpha-data /opt/glassalpha/data ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

**Data handling policies:**
```yaml
# data-handling-policy.yaml
data_governance:
  classification:
    public: []
    internal: [model_performance, system_metrics]
    confidential: [audit_reports, explanations]
    restricted: [raw_data, pii_data]

  retention:
    raw_data: 90_days
    audit_reports: 7_years
    logs: 1_year
    temporary_files: 24_hours

  access_controls:
    raw_data: [data_scientist, compliance_officer]
    audit_reports: [compliance_officer, auditor, legal]
    system_logs: [system_admin, security_officer]
```

### Audit Logging

**Comprehensive logging configuration:**
```python
# logging-config.py
import logging
import logging.handlers
import json
from datetime import datetime

class AuditFormatter(logging.Formatter):
    """Custom formatter for audit logs."""

    def format(self, record):
        # Structured logging for compliance
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
            "user": getattr(record, 'user', 'system'),
            "action": getattr(record, 'action', 'unknown'),
            "resource": getattr(record, 'resource', 'unknown'),
            "session_id": getattr(record, 'session_id', 'none'),
            "ip_address": getattr(record, 'ip_address', 'local')
        }
        return json.dumps(log_entry)

# Configure audit logging
audit_logger = logging.getLogger('glassalpha.audit')
handler = logging.handlers.RotatingFileHandler(
    '/opt/glassalpha/logs/audit.log',
    maxBytes=100*1024*1024,  # 100MB
    backupCount=10
)
handler.setFormatter(AuditFormatter())
audit_logger.addHandler(handler)
```

## Monitoring and Maintenance

### Health Monitoring

**System health checks:**
```bash
#!/bin/bash
# health-check.sh

# Check service status
systemctl is-active glassalpha >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: GlassAlpha service is not running"
    exit 1
fi

# Check disk space
DISK_USAGE=$(df /opt/glassalpha | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "WARNING: Disk usage is ${DISK_USAGE}%"
fi

# Check log file sizes
LOG_SIZE=$(du -sm /opt/glassalpha/logs | cut -f1)
if [ $LOG_SIZE -gt 1000 ]; then
    echo "WARNING: Log directory size is ${LOG_SIZE}MB"
fi

# Test basic functionality
glassalpha list >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: GlassAlpha CLI is not responding"
    exit 1
fi

echo "System health: OK"
```

**Performance monitoring:**
```python
# monitoring.py
import psutil
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_processes: int
    uptime_seconds: float

class PerformanceMonitor:
    """Monitor system performance for audit operations."""

    def __init__(self):
        self.logger = logging.getLogger('glassalpha.monitoring')

    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/opt/glassalpha').percent,
            active_processes=len(psutil.pids()),
            uptime_seconds=time.time() - psutil.boot_time()
        )

    def log_metrics(self):
        """Log system metrics for monitoring."""
        metrics = self.collect_metrics()
        self.logger.info("System metrics", extra={
            'action': 'metrics_collection',
            'metrics': metrics.__dict__
        })
```

### Backup and Recovery

**Automated backup strategy:**
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/opt/glassalpha/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="glassalpha_backup_${TIMESTAMP}.tar.gz"

echo "Starting backup at $(date)"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup critical directories
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    --exclude='/opt/glassalpha/app/venv' \
    --exclude='/opt/glassalpha/temp/*' \
    --exclude='/opt/glassalpha/logs/*.log.*' \
    /opt/glassalpha/config \
    /opt/glassalpha/data \
    /opt/glassalpha/output \
    /opt/glassalpha/.env

# Verify backup
if tar -tzf "$BACKUP_DIR/$BACKUP_FILE" >/dev/null 2>&1; then
    echo "Backup completed successfully: $BACKUP_FILE"

    # Clean old backups (keep 30 days)
    find "$BACKUP_DIR" -name "glassalpha_backup_*.tar.gz" -mtime +30 -delete
else
    echo "ERROR: Backup verification failed"
    exit 1
fi
```

**Recovery procedures:**
```bash
#!/bin/bash
# restore.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

BACKUP_FILE="$1"
RESTORE_DIR="/opt/glassalpha_restore"

echo "Starting restore from $BACKUP_FILE"

# Create restore directory
sudo mkdir -p "$RESTORE_DIR"
cd "$RESTORE_DIR"

# Extract backup
sudo tar -xzf "$BACKUP_FILE"

echo "Backup extracted to $RESTORE_DIR"
echo "Manual verification and service restart required"
```

## Scaling and Performance

### Horizontal Scaling

**Load balancer configuration (nginx):**
```nginx
# /etc/nginx/sites-available/glassalpha
upstream glassalpha_backend {
    server 10.0.1.10:8080 weight=3;
    server 10.0.1.11:8080 weight=3;
    server 10.0.1.12:8080 weight=2;
}

server {
    listen 80;
    server_name audit.company.com;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-Content-Type-Options "nosniff";
    add_header X-XSS-Protection "1; mode=block";

    location / {
        proxy_pass http://glassalpha_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeout settings for long-running audits
        proxy_read_timeout 300s;
        proxy_connect_timeout 30s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://glassalpha_backend/health;
        access_log off;
    }
}
```

### Performance Optimization

**Production performance tuning:**
```yaml
# performance-config.yaml
performance:
  # Memory management
  max_memory_per_audit: 4096  # MB
  memory_cleanup_threshold: 0.8

  # CPU optimization
  n_jobs: -1  # Use all available cores
  batch_processing: true
  batch_size: 1000

  # I/O optimization
  use_ssd_temp: true
  compression_level: 6
  async_io: true

  # Caching
  enable_result_cache: true
  cache_ttl: 3600  # 1 hour
  max_cache_size: 1024  # MB
```

**Resource monitoring and alerts:**
```python
# alerts.py
class ResourceMonitor:
    """Monitor resource usage and trigger alerts."""

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.logger = logging.getLogger('glassalpha.alerts')

    def check_resources(self) -> Dict[str, Any]:
        """Check resource usage against thresholds."""
        alerts = []

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.thresholds.get('cpu', 80):
            alerts.append({
                'type': 'cpu_high',
                'value': cpu_percent,
                'threshold': self.thresholds['cpu']
            })

        # Memory usage
        memory = psutil.virtual_memory()
        if memory.percent > self.thresholds.get('memory', 80):
            alerts.append({
                'type': 'memory_high',
                'value': memory.percent,
                'threshold': self.thresholds['memory']
            })

        # Disk usage
        disk = psutil.disk_usage('/opt/glassalpha')
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > self.thresholds.get('disk', 85):
            alerts.append({
                'type': 'disk_high',
                'value': disk_percent,
                'threshold': self.thresholds['disk']
            })

        if alerts:
            self.logger.warning("Resource alerts triggered", extra={
                'action': 'resource_alert',
                'alerts': alerts
            })

        return alerts
```

## Enterprise Deployment

### License Management

**Enterprise license configuration:**
```bash
# /opt/glassalpha/config/enterprise.env
export GLASSALPHA_LICENSE_KEY="your-enterprise-license-key"
export GLASSALPHA_LICENSE_SERVER="https://license.glassalpha.com"
export GLASSALPHA_TELEMETRY_ENABLED="false"  # Disable for air-gapped environments

# Advanced enterprise features
export GLASSALPHA_RBAC_ENABLED="true"
export GLASSALPHA_AUDIT_RETENTION_YEARS="7"
export GLASSALPHA_ADVANCED_MONITORING="true"
```

**License validation service:**
```python
# license-service.py
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional

class LicenseValidator:
    """Validate enterprise license status."""

    def __init__(self):
        self.license_key = os.getenv('GLASSALPHA_LICENSE_KEY')
        self.license_server = os.getenv('GLASSALPHA_LICENSE_SERVER')
        self.cache = {}

    def validate_license(self) -> Dict[str, Any]:
        """Validate license with caching."""
        if not self.license_key:
            return {'valid': False, 'reason': 'No license key provided'}

        # Check cache first (24 hour TTL)
        cache_key = f"license_{self.license_key[:8]}"
        cached_result = self.cache.get(cache_key)

        if cached_result and cached_result['expires'] > datetime.utcnow():
            return cached_result['result']

        # Validate with license server (if available)
        try:
            response = requests.post(
                f"{self.license_server}/validate",
                json={'license_key': self.license_key},
                timeout=10
            )
            result = response.json()

            # Cache successful validation
            if result.get('valid'):
                self.cache[cache_key] = {
                    'result': result,
                    'expires': datetime.utcnow() + timedelta(hours=24)
                }

            return result

        except Exception as e:
            # Fallback for air-gapped environments
            return self._offline_validation()

    def _offline_validation(self) -> Dict[str, Any]:
        """Offline license validation for air-gapped environments."""
        # Implement offline license validation logic
        return {'valid': True, 'mode': 'offline', 'features': ['all']}
```

### Role-Based Access Control

**RBAC configuration:**
```yaml
# rbac-config.yaml
roles:
  admin:
    permissions:
      - system_configure
      - user_manage
      - audit_all
      - report_all

  compliance_officer:
    permissions:
      - audit_run
      - audit_view
      - report_generate
      - report_export

  auditor:
    permissions:
      - audit_view
      - report_view
      - report_export

  data_scientist:
    permissions:
      - audit_run
      - model_configure
      - data_process

users:
  - username: compliance_admin
    role: admin
    email: compliance@company.com

  - username: audit_manager
    role: compliance_officer
    departments: [risk, legal]

  - username: external_auditor
    role: auditor
    temporary: true
    expires: 2024-12-31
```

## Troubleshooting

### Common Production Issues

**Service won't start:**
```bash
# Check service status
systemctl status glassalpha

# Check logs
journalctl -u glassalpha -f

# Verify permissions
ls -la /opt/glassalpha/
sudo -u glassalpha glassalpha --version

# Check Python environment
sudo -u glassalpha /opt/glassalpha/app/venv/bin/python --version
```

**Memory issues:**
```bash
# Monitor memory usage
top -u glassalpha
htop -u glassalpha

# Check for memory leaks
ps aux | grep glassalpha
pmap -x $(pidof glassalpha)

# Adjust configuration
echo "performance.max_memory_per_audit: 2048" >> config/production.yaml
```

**Performance problems:**
```bash
# Profile audit execution
time glassalpha audit --config config.yaml --output test.pdf

# Check I/O usage
iotop -u glassalpha

# Monitor system resources during audit
dstat -cdngy 5
```

### Emergency Procedures

**Service recovery:**
```bash
#!/bin/bash
# emergency-restart.sh

echo "Emergency GlassAlpha service recovery initiated"

# Stop service gracefully
sudo systemctl stop glassalpha
sleep 5

# Force kill if necessary
sudo pkill -f glassalpha

# Clear temporary files
sudo rm -rf /opt/glassalpha/temp/*

# Start service
sudo systemctl start glassalpha

# Verify recovery
sleep 10
systemctl is-active glassalpha
glassalpha list

echo "Service recovery complete"
```

## Compliance and Audit

### Audit Trail Requirements

**Complete audit logging:**
```python
# audit-trail.py
class AuditTrail:
    """Comprehensive audit trail for compliance."""

    def log_audit_start(self, config_hash: str, user: str, session_id: str):
        """Log audit initiation."""
        self.logger.info("Audit started", extra={
            'action': 'audit_start',
            'user': user,
            'session_id': session_id,
            'config_hash': config_hash,
            'timestamp': datetime.utcnow().isoformat()
        })

    def log_data_access(self, dataset_path: str, data_hash: str, user: str):
        """Log data access for compliance."""
        self.logger.info("Data accessed", extra={
            'action': 'data_access',
            'user': user,
            'dataset_path': dataset_path,
            'data_hash': data_hash,
            'timestamp': datetime.utcnow().isoformat()
        })

    def log_audit_complete(self, manifest: Dict, output_path: str, user: str):
        """Log audit completion."""
        self.logger.info("Audit completed", extra={
            'action': 'audit_complete',
            'user': user,
            'output_path': output_path,
            'manifest_hash': hash(str(manifest)),
            'timestamp': datetime.utcnow().isoformat()
        })
```

### Regulatory Reporting

**Compliance report generation:**
```bash
#!/bin/bash
# generate-compliance-report.sh

REPORT_DATE=$(date +%Y-%m-%d)
REPORT_DIR="/opt/glassalpha/compliance-reports"
mkdir -p "$REPORT_DIR"

echo "Generating compliance report for $REPORT_DATE"

# System access report
echo "=== User Access Report ===" > "$REPORT_DIR/access-report-$REPORT_DATE.txt"
grep "user_login\|data_access\|audit_start" /opt/glassalpha/logs/audit.log >> "$REPORT_DIR/access-report-$REPORT_DATE.txt"

# Audit trail report
echo "=== Audit Trail Report ===" > "$REPORT_DIR/audit-trail-$REPORT_DATE.txt"
grep "audit_complete" /opt/glassalpha/logs/audit.log >> "$REPORT_DIR/audit-trail-$REPORT_DATE.txt"

# System health report
echo "=== System Health Report ===" > "$REPORT_DIR/health-report-$REPORT_DATE.txt"
systemctl status glassalpha >> "$REPORT_DIR/health-report-$REPORT_DATE.txt"
df -h /opt/glassalpha >> "$REPORT_DIR/health-report-$REPORT_DATE.txt"

echo "Compliance reports generated in $REPORT_DIR"
```

---

## Quick Reference

**Essential Commands:**
```bash
# Service management
sudo systemctl start|stop|restart|status glassalpha

# Health checks
glassalpha --version
glassalpha list
./health-check.sh

# Configuration validation
glassalpha validate --config config/production.yaml --strict

# Run production audit
glassalpha audit --config config/production.yaml --output output/audit.pdf --strict

# View logs
tail -f /opt/glassalpha/logs/glassalpha.log
tail -f /opt/glassalpha/logs/audit.log
```

This production deployment guide is meant to help GlassAlpha operate securely and reliably in regulated industry environments while maintaining complete audit trails and compliance requirements.
