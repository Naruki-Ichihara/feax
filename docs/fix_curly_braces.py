#!/usr/bin/env python3
"""Fix curly braces in f-strings within code blocks to prevent MDX errors."""

import re
import sys
from pathlib import Path

def fix_curly_braces_in_file(file_path):
    """Escape curly braces in f-strings within code blocks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match content within code blocks (between ```python and ```)
    code_block_pattern = r'```python\n(.*?)```'

    def fix_code_block(match):
        code = match.group(1)
        # Pattern to match f-string expressions like {variable} or {expression}
        # Match {anything} but skip already escaped ones like {`something`}
        fstring_pattern = r'\{(?!`)(.*?)(?<!`)\}'

        def escape_braces(m):
            content = m.group(1)
            # Wrap in backticks to escape
            return f'{{`{content}`}}'

        # Apply the fix
        fixed_code = re.sub(fstring_pattern, escape_braces, code)
        return f'```python\n{fixed_code}```'

    # Apply to all code blocks
    new_content = re.sub(code_block_pattern, fix_code_block, content, flags=re.DOTALL)

    # Only write if content changed
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    """Process all markdown files in docs/api/reference."""
    api_dir = Path('docs/api/reference')

    if not api_dir.exists():
        print(f"Directory {api_dir} not found")
        sys.exit(1)

    # Find all .md files recursively
    md_files = list(api_dir.rglob('*.md'))

    changed_count = 0
    for md_file in md_files:
        if fix_curly_braces_in_file(md_file):
            changed_count += 1
            print(f"Fixed curly braces in: {md_file}")

    print(f"\nProcessed {len(md_files)} files, modified {changed_count} files")

if __name__ == '__main__':
    main()
