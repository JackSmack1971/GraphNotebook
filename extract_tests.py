import re
import os

def run():
    with open("test-audit.md", "r", encoding="utf-8") as f:
        content = f.read()

    # Find the python code block
    match = re.search(r'<pytest_code language="python">(.*?)</pytest_code>', content, re.DOTALL)
    if not match:
        print("No <pytest_code> block found.")
        return
    code_content = match.group(1).strip()
    
    current_file = None
    file_lines = []
    
    for line in code_content.splitlines():
        if line.startswith("# tests/") and "—" in line:
            # We hit a new file declaration
            # Finish previous file
            if current_file:
                with open(current_file, "w", encoding="utf-8") as out:
                    out.write("\n".join(file_lines).strip() + "\n")
                print(f"Created {current_file}")
            
            # Start new file
            current_file = line.replace("# ", "").split("—")[0].strip()
            file_lines = []
            
        elif line.startswith("# ============================================================================="):
            # Skip dividers
            continue
        else:
            if current_file:
                file_lines.append(line)

    # Wrap up last file
    if current_file:
        with open(current_file, "w", encoding="utf-8") as out:
            out.write("\n".join(file_lines).strip() + "\n")
        print(f"Created {current_file}")

if __name__ == "__main__":
    run()
