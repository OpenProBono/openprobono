import sys

import evaluations


def main():
    url = sys.argv[1]
    evaluations.make_questionset(url)

if __name__ == "__main__":
    main()
