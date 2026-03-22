#!/bin/bash
# Allow established connections
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow DNS
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# Allow Anthropic API (Claude Code auth + model calls)
iptables -A OUTPUT -d api.anthropic.com -j ACCEPT
iptables -A OUTPUT -d statsig.anthropic.com -j ACCEPT

# Allow Anthropic auth/OAuth endpoints
iptables -A OUTPUT -d auth.anthropic.com -j ACCEPT
iptables -A OUTPUT -d claude.ai -j ACCEPT
iptables -A OUTPUT -d anthropic.com -j ACCEPT

# Allow npm registry
iptables -A OUTPUT -d registry.npmjs.org -j ACCEPT
iptables -A OUTPUT -d npmjs.org -j ACCEPT

# Allow GitHub (for git operations)
iptables -A OUTPUT -d github.com -j ACCEPT
iptables -A OUTPUT -d api.github.com -j ACCEPT

# Block everything else outbound
iptables -A OUTPUT -j DROP