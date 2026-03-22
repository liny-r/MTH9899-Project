#!/bin/bash

cat > /etc/tinyproxy/tinyproxy.conf << EOF
Port 8888
Listen 127.0.0.1
Timeout 600
Allow 127.0.0.1
AddHeader "Connection" "close"
FilterURLs On
Filter "/etc/tinyproxy/filter"
EOF

cat > /etc/tinyproxy/filter << EOF
api\.anthropic\.com
statsig\.anthropic\.com
auth\.anthropic\.com
claude\.ai
anthropic\.com
registry\.npmjs\.org
npmjs\.org
github\.com
api\.github\.com
pypi\.org
files\.pythonhosted\.org
EOF

tinyproxy -c /etc/tinyproxy/tinyproxy.conf

# Export for all processes
echo 'export HTTPS_PROXY=http://127.0.0.1:8888' >> /etc/environment
echo 'export HTTP_PROXY=http://127.0.0.1:8888' >> /etc/environment