#!/usr/bin/env python3
"""Post-process API documentation to format parameters as bullet lists."""

import re
import sys
from pathlib import Path

def format_parameters_in_file(file_path):
    """Convert parameter sections to bullet point format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match Parameters/Arguments/Returns sections
    # Matches: "Parameters\n----------\n" followed by parameter definitions
    pattern = r'(Parameters|Arguments|Returns|Attributes|Raises)\n-+\n((?:(?!\n#{1,4}\s|\n[A-Z][a-z]+\n-+)[\s\S])*?)(?=\n#{1,4}\s|\n[A-Z][a-z]+\n-+|\Z)'

    def format_section(match):
        header = match.group(1)
        section_content = match.group(2).strip()

        if not section_content:
            return f'{header}\n{"-" * len(header)}\n\n'

        # Parse parameter definitions
        # Pattern: param_name : type
        #     Description...
        param_pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+?)$\n((?:^[ ]{4,}.*$\n?)*)'

        params = []
        for param_match in re.finditer(param_pattern, section_content, re.MULTILINE):
            param_name = param_match.group(1)
            param_type = param_match.group(2).strip()
            param_desc = param_match.group(3)

            # Clean up description - remove leading spaces
            if param_desc:
                desc_lines = param_desc.split('\n')
                cleaned_lines = []
                for line in desc_lines:
                    if line.strip():
                        # Remove leading spaces (usually 4)
                        cleaned_line = line[4:] if len(line) > 4 and line[:4] == '    ' else line.strip()
                        cleaned_lines.append(cleaned_line)
                param_desc = ' '.join(cleaned_lines).strip()
            else:
                param_desc = ''

            params.append((param_name, param_type, param_desc))

        # If no parameters were parsed, return original
        if not params:
            return match.group(0)

        # Build bullet list
        result = f'{header}\n{"-" * len(header)}\n'
        for param_name, param_type, param_desc in params:
            if param_desc:
                result += f'- **{param_name}** (*{param_type}*): {param_desc}\n'
            else:
                result += f'- **{param_name}** (*{param_type}*)\n'

        result += '\n'
        return result

    # Apply the transformation
    new_content = re.sub(pattern, format_section, content)

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
        if format_parameters_in_file(md_file):
            changed_count += 1
            print(f"Formatted parameters in: {md_file}")

    print(f"\nProcessed {len(md_files)} files, modified {changed_count} files")

if __name__ == '__main__':
    main()
