import bibtexparser
import csv
import os
import glob
import re
import argparse
import pandas as pd

# --- Constants ---
BIBTEX_DIR = 'bibtex'
PAPERS_DIR = 'papers'
DB_FILE = 'paper_database.parquet'
LOG_FILE = 'processed_files.log'

# --- 1. BibTeX -> CSV Conversion Module ---

def process_bib_file(bib_path, csv_path, conference_name):
    """
    Converts a single .bib file to a .csv file.
    conference_name is extracted from the folder name (e.g., 'cikm', 'sigir').
    """
    # [MODIFIED] Added 'Author'
    csv_headers = ['Title', 'Author', 'Conference Name (Book Title)', 'Year', 'Abstract', 'url', 'Keywords']
    
    try:
        with open(bib_path, 'r', encoding='utf-8') as bibfile:
            bib_database = bibtexparser.load(bibfile)

        with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            
            for entry in bib_database.entries:
                
                display_conf_name = conference_name.upper()

                # [MODIFIED] Added entry.get('author', '')
                row = [
                    entry.get('title', ''),
                    entry.get('author', ''),
                    display_conf_name,
                    entry.get('year', ''),
                    entry.get('abstract', ''),
                    entry.get('url', ''),
                    entry.get('keywords', '')
                ]
                writer.writerow(row)
        
        return len(bib_database.entries)

    except FileNotFoundError:
        print(f"  [Error] Input file not found: '{bib_path}'")
    except Exception as e:
        print(f"  [Error] An error occurred while processing '{bib_path}': {e}")
    return 0

def sync_csv_files(force_rebuild=False):
    """
    Syncs all .bib files under bibtex/ to papers/.
    Only overwrites files if the .bib file is newer.
    """
    print("\n--- Step 1: Syncing CSV files ---")
    
    bib_files = glob.glob(os.path.join(BIBTEX_DIR, '**/*.bib'), recursive=True)
    
    if not bib_files:
        print(f"No .bib files found in '{BIBTEX_DIR}' directory.")
        return

    total_processed = 0
    total_updated = 0
    
    for bib_path in bib_files:
        rel_path = os.path.relpath(bib_path, BIBTEX_DIR)
        
        conf_name = os.path.normpath(rel_path).split(os.sep)[0]
        
        csv_rel_path = os.path.splitext(rel_path)[0] + '.csv'
        csv_path = os.path.join(PAPERS_DIR, csv_rel_path)
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        update_needed = False
        if force_rebuild:
            update_needed = True
        elif not os.path.exists(csv_path):
            update_needed = True
        else:
            try:
                if os.path.getmtime(bib_path) > os.path.getmtime(csv_path):
                    update_needed = True
            except FileNotFoundError:
                update_needed = True 
        
        if update_needed:
            print(f"Processing: {bib_path} -> {csv_path}")
            count = process_bib_file(bib_path, csv_path, conf_name)
            if count > 0:
                print(f"  -> Saved {count} entries.")
                total_updated += 1
        
        total_processed += 1
            
    print(f"CSV sync complete: Checked {total_processed} .bib files, {total_updated} updated.")

# --- 2. Database Generation Module ---

def sync_database(force_rebuild=False):
    """
    Reads all CSVs under papers/ and builds/updates the parquet database.
    """
    print("\n--- Step 2: Syncing database ---")
    
    if force_rebuild:
        print("Force rebuild mode: Deleting existing database and log files.")
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

    all_csv_files = set(glob.glob(os.path.join(PAPERS_DIR, '**/*.csv'), recursive=True))
    
    processed_files = set()
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                processed_files = set(line.strip() for line in f)
        except Exception as e:
            print(f"Warning: Failed to read {LOG_FILE}. Rebuild recommended. ({e})")

    new_files = list(all_csv_files - processed_files)
    
    if not new_files:
        print("No new CSV files found. Database is up to date.")
        return
        
    print(f"Processing {len(new_files)} new CSV files...")
        
    df_list = []
    for f in new_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: Failed to load file {f}. {e}")
            
    if not df_list:
        print("Error: No valid data to process.")
        return

    new_df = pd.concat(df_list, ignore_index=True)
    new_df['Title'] = new_df['Title'].fillna('')
    new_df['Author'] = new_df['Author'].fillna('')
    new_df['Abstract'] = new_df['Abstract'].fillna('')
    new_df['Keywords'] = new_df['Keywords'].fillna('')

    if os.path.exists(DB_FILE):
        print("Loading and merging with existing database.")
        try:
            old_df = pd.read_parquet(DB_FILE)
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"Error: Failed to load existing data. Overwriting. ({e})")
            combined_df = new_df
    else:
        print("Creating new database file.")
        combined_df = new_df

    try:
        combined_df.to_parquet(DB_FILE, index=False)
        
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            for file_path in new_files:
                f.write(f"{file_path}\n")
        
        print(f"\nDatabase sync complete: Total {len(combined_df)} papers saved.")
        
    except Exception as e:
        print(f"Error: Failed to save final database. {e}")

# --- 3. Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description='Converts BibTeX to CSV and incrementally updates the embedding database.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Ignore modification times, force CSV conversion, and rebuild the entire embedding database.'
    )
    args = parser.parse_args()

    sync_csv_files(force_rebuild=args.force)
    sync_database(force_rebuild=args.force)
    
    print("\n[Complete] All update processes have finished.")

if __name__ == "__main__":
    main()