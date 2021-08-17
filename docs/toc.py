#! /usr/bin/env python3

# Output a table of contents for the given Markdown file.

from sys import argv

md_file = open(argv[1])

def slug(a_string):
    for char in './+-()': # and other url-unfriendly chars...
        a_string = a_string.replace(char, ' ')
    a_string = a_string.lower()
    return '-'.join(a_string.split()) 

in_code_block = False
for line in md_file:
    if line.startswith('```'):
        in_code_block = not in_code_block
    if line.startswith('#') and not in_code_block:
        level, headline = line.split(' ', 1)
        level = level.count('#')
        headline = headline.rstrip('# \n')
        print('{}- [{}](#{})'.format(
            '  ' * (level - 1), headline, slug(headline)))

