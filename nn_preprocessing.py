from Data_conversion.extract_13_regions_from_CKI import nearest_neighbour_process
import time


def main():
    input_path = 'data/nnunet/raw/Dataset018_Orig_nn/labelsTr'
    output_path = 'data/nnunet/raw/Dataset018_Orig_nn/'
    nearest_neighbour_process(input_path, output_path)


if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"--- {total_time:.2f} seconds ---")