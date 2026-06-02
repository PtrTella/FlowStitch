#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix the _build_consolidated_notebooks.py script syntax issues."""
import ast

path = '/Users/tella/Workspace/FlowStitch/notebooks/_build_consolidated_notebooks.py'
content = open(path, 'r').read()

# Replace print(""" and print(f""" with print(''' and print(f'''
fixed = content.replace('print("""', "print('''").replace('print(f"""', "print(f'''")

# Fix closing triple-quotes: lines that are just """) after a print(''' block
lines = fixed.split('\n')
in_print_triple = False
result_lines = []
for line in lines:
    if "print('''" in line or "print(f'''" in line:
        in_print_triple = True
    if in_print_triple and line.strip() == '""")':
        line = line.replace('""")', "''')")
        in_print_triple = False
    result_lines.append(line)

fixed = '\n'.join(result_lines)
open(path, 'w').write(fixed)

try:
    ast.parse(fixed)
    print('Syntax OK')
except SyntaxError as e:
    arr = fixed.split('\n')
    print(f'SyntaxError at line {e.lineno}: {e.msg}')
    for i in range(max(0, e.lineno-3), min(len(arr), e.lineno+3)):
        print(f'  {i+1}: {arr[i][:120]}')
