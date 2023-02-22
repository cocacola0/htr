import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--width', type=int, default = 128)
    parser.add_argument('--height', type=int, default = 32)

    args = parser.parse_args()
    return args