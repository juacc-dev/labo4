import re

s = '26-08-(2020)'

grp = re.search(r'([\d]{2})-([\d]{2})-\(([\d]{4})\)', s)
print(grp.groups())
