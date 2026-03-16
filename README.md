# RecSys Paper Finder

RecSys Paper Finder is an open, non-commercial project designed to make it easy to search and filter academic papers, with a focus on the Recommender Systems (RecSys) domain.

Live deployment: https://recsys-paper-finder.streamlit.app/

Users can build a paper database from their own BibTeX (.bib) files and leverage natural language processing for keyword-based and ranking-based search to find relevant papers.

## Key Features

- Semantic Discovery and Ranking: Uses the BM25Okapi algorithm (BM25) to effectively rank papers by relevance to search queries.
- Keyword Filtering: Supports strict matching for specific keywords with AND/OR filtering logic.
- Year Filtering: Easily narrow down paper results by publication year.
- Automated Processing: The update.py script automatically converts .bib files into .csv format and updates the central database.
- RecSys Focus: Automatically classifies search results into "RecSys Papers" and "Other Papers" using domain-specific keyword heuristics.
- Analytics View: Visualizes the paper library by conference and publication year using Altair.

## UI Overview

The user interface of the project is organized as follows.

### 1. Main Screen and Database Sync

![Main UI](docs/images/main_ui.png)

When you launch the app, you will first see the "Database Sync & Maintenance" option.
This section allows you to easily update the paper database based on the latest BibTeX files.

### 2. Search and Results Screen

![Search Results](docs/images/search_results.png)

In the "Define Search Strategy" area, you can choose from three primary search modes:
- Semantic Discovery (BM25)
- Exact Keyword (Exact)
- Author Search (Author)

You can also use the "Refine Results" area to narrow down results by specifying a particular conference or publication year range.
Search results are displayed across "RecSys Papers", "Potential RecSys", and "Other Papers" tabs, allowing you to quickly browse papers of interest.

## Project Structure

```text
recsys-paper-finder/
├── app.py                      # Main Streamlit web application
├── update.py                   # Script that processes bibtex files and builds the search database
├── requirements.txt            # Python dependency list
├── bibtex/                     # [User Input] Directory for storing original .bib files
├── papers/                     # [Auto-generated] Directory for converted .csv files
└── paper_database.parquet      # [Auto-generated] Compiled database of all processed papers
```

## Setup and Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/recsys-paper-finder.git
cd recsys-paper-finder
```

**2. Install dependencies**
Make sure Python 3.8 or higher is installed, then install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Add BibTeX Files
Place your .bib files under the `bibtex/` directory using the following structure:

```
bibtex/<conference_name>/<conference_name><year>.bib
```

Each conference should have its own subdirectory, and each .bib file should be named with the conference name followed by the year. For example:
```text
bibtex/
  ├── sigir/
  │   ├── sigir2023.bib
  │   └── sigir2024.bib
  └── kdd/
      ├── kdd2022.bib
      └── kdd2023.bib
```
The first-level folder name under `bibtex/` (e.g., "sigir", "kdd") is used as the conference name. The .bib filenames must follow the `<conference_name><year>.bib` pattern.

### 2. Build the Database
Run the update script to parse the BibTeX files, convert them to CSV format, and build the database:
```bash
python update.py
```
To rebuild the entire database from scratch, use the `--force` flag:
```bash
python update.py --force
```

### 3. Run the App
Start the Streamlit interface:
```bash
streamlit run app.py
```
This will open the user interface in a web browser (typically at `http://localhost:8501`).

## Dependencies
- `streamlit` - Web interface.
- `pandas` & `pyarrow` - Data processing and `.parquet` file storage.
- `rank_bm25` - BM25Okapi search ranking algorithm.
- `altair` - Visualization of paper count metrics.
- `bibtexparser` - Parsing `.bib` files.