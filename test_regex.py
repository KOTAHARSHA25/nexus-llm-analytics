import re

models = ['llama2:13b', 'command-r:35b', 'llama3:3b', 'phi3:mini', 'qwen2:1.5b', 'tinyllama:latest', 'llama2:7b', 'llama2:70b']

# The pattern should match :Xb where X starts with 0-3
# So :0.5b, :1b, :1.5b, :2b, :3b YES
# But :13b, :35b, :7b, :70b NO
pattern = r'[:_\-\s]([0-3](?:\.\d+)?)\s*b\b'

for m in models:
    match = re.search(pattern, m.lower())
    print(f'{m}: {bool(match)} - matched: {match.group(0) if match else "N/A"}')
