
import os

log_file = r'logs\nexus.log'
if os.path.exists(log_file):
    with open(log_file, 'rb') as f:
        try:
            f.seek(-4000, 2)
        except OSError:
            f.seek(0)
        content = f.read()
        print(content.decode('utf-8', errors='replace'))
else:
    print(f"Log file not found at {log_file}")
