import socket
import sys

def check_server_auth(allowed_hostnames):
    current_hostname = socket.gethostname()
    if current_hostname not in allowed_hostnames:
        print(f"Unauthorized host: {current_hostname}")
        sys.exit(1)
    print(f"Authorized host: {current_hostname}")
