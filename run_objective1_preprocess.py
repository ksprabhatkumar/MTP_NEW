"""Build the compact Objective 1 REALDISP smoke-test data."""

from process_all_data import process_realdisp


if __name__ == "__main__":
    process_realdisp({2, 5, 15})
