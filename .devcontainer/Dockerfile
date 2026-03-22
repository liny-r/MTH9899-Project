FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Install firewall tools if not running in docker
# Must edit one line of devcontainer.json with: 
# "postStartCommand": "sudo /usr/local/bin/init-firewall.sh",
# RUN apt-get install -y iptables

# Install firewall tools for docker devcontainer
RUN apt-get install -y tinyproxy 

# Needed for firewall 

# Install Python packages
RUN apt-get install -y python3-pip python3-venv 
RUN pip3 install --break-system-packages \
    ipykernel \
    numpy \
    pandas \
    jupyter \ 
    Joblib \ 
    statsmodels \
    tqdm \
    scipy \
    scikit-learn \
    plotly \
    python-dotenv \
    pytest  

# Copy and enable firewall script for non-docker
# COPY init-firewall.sh /usr/local/bin/init-firewall.sh
# RUN chmod +x /usr/local/bin/init-firewall.sh