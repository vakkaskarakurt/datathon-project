import os
import fnmatch

def parse_gitignore(gitignore_path):
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    return patterns

def is_ignored(path, patterns):
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False

def create_project_structure_txt(root_dir, output_file):
    gitignore_path = os.path.join(root_dir, '.gitignore')
    ignore_patterns = parse_gitignore(gitignore_path)

    with open(output_file, 'w') as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            rel_dir = os.path.relpath(dirpath, root_dir)
            if rel_dir == '.':
                rel_dir = ''
            if is_ignored(rel_dir + '/', ignore_patterns):
                continue
            level = dirpath.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f'{indent}{os.path.basename(dirpath)}/\n')
            subindent = ' ' * 4 * (level + 1)
            # Filter ignored directories
            dirnames[:] = [d for d in dirnames if not is_ignored(os.path.join(rel_dir, d) + '/', ignore_patterns)]
            for filename in filenames:
                rel_file = os.path.join(rel_dir, filename)
                if filename == '.gitignore' or is_ignored(rel_file, ignore_patterns):
                    continue
                f.write(f'{subindent}{filename}\n')

create_project_structure_txt('.', 'project_structure.txt')
