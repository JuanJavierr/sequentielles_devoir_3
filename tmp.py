def generate_labyrinth(width, height):
    labyrinth = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(0)
        labyrinth.append(row)
    return labyrinth


def print_labyrinth(labyrinth):
    for row in labyrinth:
        print(row)


def main():
    labyrinth = generate_labyrinth(10, 10)
    print_labyrinth(labyrinth)


if __name__ == "__main__":
    main()
