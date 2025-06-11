#!/bin/bash

# Check if VM details are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <vm_username> <vm_ip> <vm_ssh_key_path>"
    exit 1
fi

VM_USERNAME=$1
VM_IP=$2
VM_SSH_KEY=$3

# Create remote directory
ssh -i "$VM_SSH_KEY" "$VM_USERNAME@$VM_IP" "mkdir -p ~/resume-coach"

# Copy files to VM
scp -i "$VM_SSH_KEY" -r \
    backend \
    frontend \
    requirements.txt \
    deploy_vm.sh \
    "$VM_USERNAME@$VM_IP:~/resume-coach/"

# Make deploy script executable
ssh -i "$VM_SSH_KEY" "$VM_USERNAME@$VM_IP" "chmod +x ~/resume-coach/deploy_vm.sh"

echo "Files copied successfully!"
echo "To deploy, SSH into the VM and run:"
echo "cd ~/resume-coach && ./deploy_vm.sh" 