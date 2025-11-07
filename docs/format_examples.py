#!/usr/bin/env python3
"""Post-process API documentation to format examples with code blocks."""

import re
import sys
from pathlib import Path

def format_examples_in_file(file_path):
    """Add Python code blocks around example sections."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match Examples/Example sections followed by lines starting with >>> (HTML-escaped as &gt;&gt;&gt;)
    # This regex finds "Examples\n--------\n" followed by example code
    pattern = r'(Examples?)\n-+\n((?:(?:&gt;&gt;&gt;|\.\.\.).*\n?)+)'

    def replace_example(match):
        header = match.group(1)
        example_code = match.group(2).strip()
        # Decode HTML entities: &gt; -> >, &lt; -> <, &amp; -> &
        example_code = example_code.replace('&gt;', '>')
        example_code = example_code.replace('&lt;', '<')
        example_code = example_code.replace('&amp;', '&')
        # Wrap in Python code block
        return f'{header}\n{"-" * len(header)}\n```python\n{example_code}\n```\n'

    # Apply the transformation
    new_content = re.sub(pattern, replace_example, content)

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
        if format_examples_in_file(md_file):
            changed_count += 1
            print(f"Formatted examples in: {md_file}")

    print(f"\nProcessed {len(md_files)} files, modified {changed_count} files")

if __name__ == '__main__':
    main()
