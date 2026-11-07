from pathlib import Path
from collections import Counter
import csv
import json
import re

ROOT = Path(r"C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825")
OUTPUT = ROOT / "attrex_inventory.csv"
TEXT_EXT = {'.txt','.ict','.csv','.tsv','.dat','.asc','.ann','.xml','.json','.nc','.h5','.hdf','.hdf5'}

def main():
    if not ROOT.is_dir():
        print('Directory not found:', ROOT)
        print('Check that the folder is available locally through OneDrive.')
        return
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and p != OUTPUT)
    print('Root:', ROOT)
    print('Files:', len(files))
    print('Extensions:', dict(Counter(p.suffix.lower() or '[none]' for p in files)))
    with OUTPUT.open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.writer(out)
        writer.writerow(['relative_path','extension','size_bytes'])
        for p in files:
            writer.writerow([str(p.relative_to(ROOT)), p.suffix.lower(), p.stat().st_size])
    print('Inventory saved:', OUTPUT)
    print('\nFIRST 30 FILES:')
    for p in files[:30]:
        print(f'{p.stat().st_size:>12,}  {p.relative_to(ROOT)}')
    print('\nSAMPLE FILE HEADERS (up to 3 candidate files):')
    candidates = [p for p in files if p.suffix.lower() in TEXT_EXT and p.stat().st_size > 0]
    for p in candidates[:3]:
        print('\n---', p.name, '---')
        ext = p.suffix.lower()
        if ext in {'.nc','.h5','.hdf','.hdf5'}:
            try:
                import xarray as xr
                ds = xr.open_dataset(p)
                print(ds)
                ds.close()
            except Exception as e:
                print('Metadata requires a compatible backend:', type(e).__name__, str(e)[:200])
        else:
            with p.open('rb') as f:
                sample = f.read(6000)
            if b'\x00' in sample[:256]:
                print('Binary content; first 64 bytes:', repr(sample[:64]))
            else:
                for line in sample.decode('utf-8', errors='replace').splitlines()[:35]:
                    print(line[:220])
    print('\nPlease share the console output and, if convenient, attrex_inventory.csv.')

if __name__ == '__main__':
    main()
