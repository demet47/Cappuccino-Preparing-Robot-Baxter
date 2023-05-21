import csv
import os

#get all files in this folder that ends with .csv
files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith(".csv")]

for file in files:
    new_rows = []

    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        row0 = next(reader)
        row0.insert (1, "x")
        row0.insert (2, "y") 
        new_rows.append(row0)   
        print(row0)
        for row in reader:
            row.insert (1, "0")
            row.insert (2, "0")
            new_rows.append(row)
            print(row)

    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(new_rows)