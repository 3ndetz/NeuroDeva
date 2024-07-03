import random
import string
from pathlib import Path

data_dir_test = Path("../images/train")

symbols = {i: 0 for i in string.ascii_lowercase + string.digits}
lengths = [0] * 100
for i in data_dir_test.glob("**/*"):
    name = i.name.split('-')[0].split('.')[0]
    for i in name:
        symbols[i] += 1
    lengths[len(name)] += 1
ans = []
# print char -> count
for i, item in sorted(symbols.items(), key=lambda e: -e[1]):
    print(item, i)
    if item > 0: ans.append(i)
print('_________')
# print length -> count
for i, item in enumerate(lengths):
    if item == 0: continue
    print(f'LEN[{i}]={item}')


print(ans)

