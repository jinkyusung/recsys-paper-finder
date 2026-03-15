# RecSys Paper Finder 🔎

**RecSys Paper Finder** is a public, non-commercial project designed to easily search and filter academic papers, focusing specifically on Recommender Systems (RecSys).

🔗 **Live Deployment:** [https://recsys-paper-finder.streamlit.app/](https://recsys-paper-finder.streamlit.app/)

Users can populate the database using their own BibTeX (`.bib`) files and leverage natural language processing to perform keyword-based and ranking-based searches to discover relevant papers.

## Features

- **Semantic Discovery & Ranking**: Uses the `BM25Okapi` algorithm (BM25) to effectively rank papers based on the relevance of your search queries.
- **Keyword Filtering**: Support for strict matching (AND/OR filtering logic) for specified keywords.
- **Year Filtering**: Easily narrow down your paper results by publication year.
- **Automated Processing**: The `update.py` script automatically converts nested `.bib` files within conference folders to `.csv` and updates the central database.
- **RecSys Focused**: Automatically categorizes search results into "Recommender System Papers" and "Other Papers" using domain-specific keyword heuristics.
- **Analytics View**: Visualizes your library by conference and publication year using Altair.

## Project Structure

```text
recsys-paper-finder/
├── app.py                      # Main Streamlit web application
├── update.py                   # Script to process bibtex files and build the search database
├── requirements.txt            # Python dependencies
├── bibtex/                     # [Your input] Directory to store source .bib files
├── papers/                     # [Auto-generated] Directory where converted .csv files are stored
└── paper_database.parquet      # [Auto-generated] Compiled database of all processed papers
```

## Setup & Installation

**1. Clone the repository and navigate into the folder**
```bash
git clone https://github.com/yourusername/recsys-paper-finder.git
cd recsys-paper-finder
```

**2. Install Dependencies**
Ensure you have Python 3.8+ installed. Then install the required packages:
```bash
pip install -r requirements.txt
```

## How to Use

### 1. Add your BibTeX files
Create a directory structure under `bibtex/` representing your conferences/journals, and place your `.bib` files there. For example:
```text
bibtex/
  ├── sigir/
  │   └── sigir_2023.bib
  ├── kdd/
  │   └── 2022/
  │       └── kdd_2022.bib
```
*Note: The script automatically uses the first-level folder names under `bibtex/` as the Conference/Book Title names (e.g., "sigir", "kdd"), regardless of how deeply nested the `.bib` files are.*

### 2. Build the Database
Run the update script to parse your BibTeX files, convert them to CSV format, and build the database:
```bash
python update.py
```
If you encounter errors or wish to rebuild the whole database from scratch, use the `--force` flag:
```bash
python update.py --force
```

### 3. Run the App
Launch the Streamlit interface:
```bash
streamlit run app.py
```
This will open the user interface in your web browser (usually at `http://localhost:8501`).

## Dependencies
- `streamlit` - For the web interface.
- `pandas` & `pyarrow` - For data processing and `.parquet` file storage.
- `rank_bm25` - For the BM25Okapi search ranking algorithm.
- `altair` - For visualizing paper count metrics.
- `bibtexparser` - For parsing the `.bib` files.