# RecSys Paper Finder 🔎

RecSys Paper Finder is a Streamlit-based web application designed to help researchers efficiently search and filter academic papers, with a particular focus on Recommender Systems (RecSys). 

Users can populate the database using their own BibTeX (`.bib`) files and leverage natural language processing to perform semantic and keyword-based searches to discover relevant papers.

## Features

- **Semantic Search**: Uses the `all-MiniLM-L6-v2` SentenceTransformer model to understand the meaning of your queries, not just exact keyword matches.
- **Keyword Filtering**: Support for strict matching (AND/OR filtering logic) for specified keywords.
- **Year Filtering**: Easily narrow down your paper results by publication year.
- **Automated Processing**: `update.py` script automatically converts `.bib` files to `.csv` and incrementally computes embeddings for semantic search.
- **RecSys Focused**: Automatically categorizes search results into "Recommender System Papers" and "Other Papers" using domain-specific heuristics and similarity to the "recsys" concept.
- **Analytics View**: Visualizes your library by conference and publication year using Altair.

## Project Structure

```text
recsys-paper-finder/
├── app.py                      # Main Streamlit web application
├── update.py                   # Script to process bibtex files and build the search database
├── requirements.txt            # Python dependencies
├── bibtex/                     # [Your input] Directory to store source .bib files
├── papers/                     # [Auto-generated] Directory where converted .csv files are stored
├── paper_database.parquet      # [Auto-generated] Compiled database of all processed papers
└── paper_embeddings.npy        # [Auto-generated] Pre-computed embeddings for fast semantic search
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
  │   └── kdd_2022.bib
```
*Note: The script uses the folder names under `bibtex/` as the Conference/Book Title names (e.g., "sigir", "kdd").*

### 2. Build the Database
Run the update script to parse your BibTeX files, convert them to CSV format, and compute the paper embeddings:
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
- `numpy` & `sentence-transformers` - For managing arrays and calculating textual similarity.
- `altair` - For visualizing paper count metrics.
- `bibtexparser` - For parsing the `.bib` files.
