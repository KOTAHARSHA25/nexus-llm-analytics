import ast
import os
import json
import glob

SOURCE_DIRS = [
    r"src/backend/plugins",
    r"src/backend/core",
    r"src/backend/api"
]

RISKY_KEYWORDS = [
    "eval", "exec", "subprocess", "open", "requests", "connect", "execute", 
    "shell", "system", "dill", "pickle"
]

def scan_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        functions = []
        classes = []
        risks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in RISKY_KEYWORDS:
                        risks.append(f"Call to {node.func.id} at line {node.lineno}")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in RISKY_KEYWORDS:
                        risks.append(f"Call to {node.func.attr} at line {node.lineno}")

        return {
            "functions": functions,
            "classes": classes,
            "critical_risks": risks
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    inventory = {}
    base_path = os.getcwd()
    
    for dir_path in SOURCE_DIRS:
        full_dir = os.path.join(base_path, dir_path)
        # Recursive glob to include subdirectories like core/engine
        files = glob.glob(os.path.join(full_dir, "**/*.py"), recursive=True)
        
        for file_path in files:
            # Normalize path relative to src
            rel_path = os.path.relpath(file_path, base_path).replace("\\", "/")
            if "__init__.py" in rel_path:
                continue
                
            inventory[rel_path] = scan_file(file_path)
            
    with open("inventory.json", "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print("Inventory saved to inventory.json")

if __name__ == "__main__":
    main()
