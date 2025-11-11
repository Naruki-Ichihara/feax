#!/usr/bin/env python3
"""Post-process API documentation to format examples with code blocks."""

import re
import sys
from pathlib import Path

def format_examples_in_file(file_path):
    """Add Python code blocks around example sections."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern 1: Match entire "Examples" section until next heading or end
    # Captures everything from "Examples\n--------" until next "####" or other section
    pattern1 = r'(Examples?)\n(-+)\n((?:(?!^#{1,4}\s|^[A-Z][a-z]+\n-+).*\n?)*?)(?=^#{1,4}\s|^[A-Z][a-z]+\n-+|\Z)'

    def replace_example_underline(match):
        header = match.group(1)
        underline = match.group(2)
        section_content = match.group(3).strip()

        if not section_content:
            return f'{header}\n{underline}\n\n'

        # Split into lines and process
        lines = section_content.split('\n')
        result_lines = []
        in_code_block = False
        code_lines = []

        for line in lines:
            # Check if line is a code line (starts with >>> or ...)
            is_code = line.lstrip().startswith(('&gt;&gt;&gt;', '...'))

            if is_code:
                if not in_code_block:
                    # Start new code block
                    in_code_block = True
                    code_lines = []
                # Decode HTML entities
                decoded_line = line.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
                code_lines.append(decoded_line)
            else:
                if in_code_block:
                    # End code block and wrap it
                    result_lines.append('```python')
                    result_lines.extend(code_lines)
                    result_lines.append('```')
                    in_code_block = False
                    code_lines = []
                # Add non-code line
                if line.strip():  # Only add non-empty lines
                    result_lines.append(line)

        # Close any remaining code block
        if in_code_block:
            result_lines.append('```python')
            result_lines.extend(code_lines)
            result_lines.append('```')

        return f'{header}\n{underline}\n' + '\n'.join(result_lines) + '\n\n'

    # Pattern 2: Match "**Example**:\n\n" format (flat/experimental modules)
    # Captures everything from **Example**: until next heading or end
    pattern2 = r'\*\*Example\*\*:\n\n((?:(?!^#{1,4}\s).*\n?)*?)(?=^#{1,4}\s|\Z)'

    def replace_example_bold(match):
        section_content = match.group(1).strip()

        if not section_content:
            return '**Example**:\n\n'

        # Split into lines and process
        lines = section_content.split('\n')
        result_lines = []
        in_code_block = False
        code_lines = []

        for line in lines:
            # Remove leading 2-space indentation if present
            dedented = line[2:] if line.startswith('  ') else line

            # Check if line is a code line (starts with >>> or ... after dedenting)
            is_code = dedented.lstrip().startswith(('&gt;&gt;&gt;', '...'))

            if is_code:
                if not in_code_block:
                    # Start new code block
                    in_code_block = True
                    code_lines = []
                # Decode HTML entities
                decoded_line = dedented.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
                code_lines.append(decoded_line)
            else:
                if in_code_block:
                    # End code block and wrap it
                    result_lines.append('```python')
                    result_lines.extend(code_lines)
                    result_lines.append('```')
                    in_code_block = False
                    code_lines = []
                # Add non-code line (descriptive text)
                if dedented.strip():  # Only add non-empty lines
                    result_lines.append(dedented)

        # Close any remaining code block
        if in_code_block:
            result_lines.append('```python')
            result_lines.extend(code_lines)
            result_lines.append('```')

        return '**Example**:\n\n' + '\n'.join(result_lines) + '\n\n'

    # Apply both transformations
    new_content = re.sub(pattern1, replace_example_underline, content, flags=re.MULTILINE)
    new_content = re.sub(pattern2, replace_example_bold, new_content, flags=re.MULTILINE)

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
