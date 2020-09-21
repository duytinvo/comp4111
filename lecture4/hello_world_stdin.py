import sys

print(f'Press q to exit!')
print("Hey, what's your name?")

for line in sys.stdin:
    if 'q' == line.rstrip():
        break
    print(f'Hello {line.strip()}!\n')
    print("Hey, what's your name?")

print("Exit")
