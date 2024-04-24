import sys
from functions import plot_unlabeled


def main():
    # Check command line arguments
    if len(sys.argv) not in [1, 2, 3]:
        sys.exit("Usage: python predict.py [model.keras] [data_directory]")
        # python predict.py model.keras gtsrb

    model = sys.argv[1] if len(sys.argv) > 1 else "model.keras"
    directory = sys.argv[2] if len(sys.argv) == 3 else "gtsrb"

    plot_unlabeled(model, directory)


if __name__ == "__main__":
    main()
