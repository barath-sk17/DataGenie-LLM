import csv
import os

def partition_csv(input_file, output_folder, rows_per_file=1000):
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read header

        row_count = 0
        file_count = 1
        output_file = os.path.join(output_folder, f'output_{file_count}.csv')
        os.makedirs(output_folder, exist_ok=True)

        for row in reader:
            if row_count % rows_per_file == 0:
                if row_count > 0:
                    output_csvfile.close()
                output_csvfile = open(output_file, 'w', newline='')
                writer = csv.writer(output_csvfile)
                writer.writerow(header)  # Write header to each file
                file_count += 1
                output_file = os.path.join(output_folder, f'output_{file_count}.csv')

            writer.writerow(row)
            row_count += 1

        if row_count % rows_per_file != 0:
            output_csvfile.close()

# Example usage:
input_file = 'SalesUseCase-V1.csv'
output_folder = 'Sales-Datagen'
partition_csv(input_file, output_folder)
