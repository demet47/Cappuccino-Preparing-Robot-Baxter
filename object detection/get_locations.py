import os
import json

folders = [f.path for f in os.scandir('./cup_place_finderv2-1') if f.is_dir()]

subs = []
for folder in folders:
    subfolders = os.scandir(folder)
    for subfolder in subfolders:
        if(subfolder.path.endswith("labels")):
            subs.append(subfolder.path)

all_files = []
for sub in subs:
    files = [sub+"\\"+f for f in os.listdir(sub) if f.endswith(".txt")]
    all_files.extend(files)

locations = []
for file in all_files:
    print(file)
    with open(file, 'r') as f:
        for line in f:
            locations.append({file[-(len(file)-file.find("table")):] : line.split()[1:]}) #-44

# Serializing json
json_object = json.dumps(locations)
 
# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)

# Gereksiz olan locationları sil ve sonra devam et. Aynı fotonun birden fazla location ı var ve fotonun ayna görüntüsünün locationı yanlış.
# dataların 4 tanesinin ilk ikisi orta noktanın locationını veriyor.
with open("sample.json", "r") as inputfile:
    inf = json.load(inputfile)

new_inf = [f[:2] for f in inf]

