import csv
import os
import re

#get all files in this folder that ends with .csv
files = [f for f in os.listdir(".") if f.endswith(".csv")]

# TODO this needs to be changed everytime. Locations of the objects in the image.
new_inf = [
    ['0.25703125', '0.76171875'],
    ['0.259375', '0.76484375'],
    ['0.17890625', '0.87734375'],
    ['0.159375', '0.63125'],
    ['0.36328125', '0.6234375'],
    ['0.37265625', '0.88828125'],
    ['0.5109375', '0.67109375'],
    ['0.3640625', '0.75703125'],
    ['0.69140625', '0.7609375'],
    ['0.57734375', '0.80859375'],
    ['0.684375', '0.66484375'],
    ['0.6671875', '0.8859375'],
    ['0.6109375', '0.7109375'],
    ['0.55390625', '0.9140625'],
    ['0.2625', '0.61328125'],
    ['0.79921875', '0.70859375'],
    ['0.80078125', '0.865625'],
    ['0.8734375', '0.775'],
    ['0.2765625', '0.88984375'],
    ['0.8953125', '0.89921875']
]


for file in files:
    new_rows = []

    insertValX = 0
    insertValY = 0

    if(len(re.split("_", file)) == 3):
        insertValX = new_inf[int(re.split("_", file)[1])-1][0]
        insertValY =  new_inf[int(re.split("_", file)[1])-1][1]

    print(insertValX)
    print(insertValY)
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        row0 = next(reader)
        row0.insert (1, "x")
        row0.insert (2, "y") 
        new_rows.append(row0)   
        print(row0)
        for row in reader:
            row.insert (1, insertValX)
            row.insert (2, insertValY)
            new_rows.append(row)
            print(row)

    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(new_rows)