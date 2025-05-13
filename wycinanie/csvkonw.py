import csv

# Ścieżki do plików
input_txt = "Subject3.txt"
output_csv = "output.csv"

# Konwersja tab-separated → comma-separated
with open(input_txt, 'r', encoding='utf-8') as infile, open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter=',')
    for row in reader:
        writer.writerow(row)

print("Zamieniono na CSV:", output_csv)
