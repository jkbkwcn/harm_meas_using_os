# Converts .npy files to .csv files
import numpy as np


def npy_to_csv(npy_file_path: str, csv_file_path: str) -> None:
    """
    Converts a NumPy .npy file to a CSV file.

    Args:
        npy_file_path (str): Path to the input .npy file.
        csv_file_path (str): Path to the output .csv file.
    """
    data = np.load(npy_file_path)
    np.savetxt(csv_file_path, data, delimiter=",")


def main():
    # Example usage
    npy_file = "measured_data/adc_data_sin_3_1Vpkpk_75Hz.npy"
    csv_file = "measured_data/adc_data_sin_3_1Vpkpk_75Hz.csv"
    npy_to_csv(npy_file, csv_file)


if __name__ == "__main__":
    main()
