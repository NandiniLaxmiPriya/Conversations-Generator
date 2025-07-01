import re

def extract_content_only(text):
    lines = text.splitlines()

    # Step 1: Skip the first 3 lines (character introductions)
    content_lines = lines[3:]

    cleaned_lines = []
    for line in content_lines:
        line = line.strip()

        # Step 2: Remove Telugu speaker names at start (e.g., "రమేష్:")
        line = re.sub(r"^[\u0C00-\u0C7F\s]+:", "", line)

        # Step 3: Trim leftover punctuation or whitespace
        line = line.strip(" -*•")

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# Example usage
with open("conversations_output.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

content_only = extract_content_only(raw_text)

with open("conversations_output_c.txt", "w", encoding="utf-8") as f:
    f.write(content_only)

print("Cleaned text saved to cleaned_output.txt")
