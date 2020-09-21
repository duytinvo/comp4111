
print(f'Press q to exit!')
while True:
    line = input("Hey, what's your name?\n")
    if 'q' == line.rstrip():
        break
    print(f'Hello {line.rstrip()}!\n')

print("Exit")