# folder_path = r'C:\Users\splpil\OneDrive - SAS\splpil\62- CA Italy - Card Fraud Model PoC\CA_framework_PoC\Raw Data\test'
import os

import pandas as pd

# Ścieżka do folderu z plikami CSV
# Path to the folder with CSV files
folder_path = (
    r"C:\Users\splpil\OneDrive - SAS\splpil\62- CA Italy - Card Fraud Model"
    r" PoC\CA_framework_PoC\Raw Data"
)

# Lista plików CSV w folderze
# List of CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# Zmienne do przechowywania sumy, liczby rekordów i liczby wczytanych plików
# Variables to store the sum, number of records and number of files loaded
total_amount = 0
total_records = 0
files_loaded = 0

# Iteracja po wszystkich plikach CSV
# Iterate over all CSV files
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    # Wczytanie danych z pliku CSV z kodowaniem latin-1
    # Loading data from a CSV file with latin-1 encoding
    df = pd.read_csv(file_path, sep=";", encoding="latin-1")
    # Dodanie sumy z kolumny 'AMOUNT_BASE' do ogólnej sumy
    # Adding the sum from the 'AMOUNT_BASE' column to the total
    total_amount += df["AMOUNT_BASE"].sum()
    # Dodanie liczby rekordów w pliku do ogólnej liczby rekordów
    # Adding the number of records in the file to the total number of records
    total_records += len(df)
    # Zwiększenie licznika wczytanych plików
    # Increase the number of loaded files
    files_loaded += 1

    # Wypisanie liczby rekordów w pliku
    # Print the number of records in the file
    print(f"Liczba rekordów w pliku {file}: {len(df)}")

# Wyświetlenie ogólnej sumy, liczby rekordów i liczby wczytanych plików
# Display the total, number of records and number of files loaded
print("Ogólna suma w kolumnie 'AMOUNT_BASE':", total_amount)
print("Ogólna liczba rekordów:", total_records)
print("Liczba wczytanych plików:", files_loaded)
