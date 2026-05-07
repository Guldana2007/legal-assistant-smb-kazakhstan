"""
ingest.py — Document indexing script
======================================
Run once to index all documents from the docs/ folder into ChromaDB.

Usage:
    python ingest.py                  # index all files in docs/
    python ingest.py report.pdf       # index a specific file
    python ingest.py report.pdf report2.docx  # index multiple files
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from agents.document_loader import load_document, list_docs, DOCS_DIR


def ingest_all():
    files = list_docs()
    if not files:
        print(f"No PDF/DOCX files found in {DOCS_DIR}")
        print("Put your documents there and run again.")
        return

    print(f"Found {len(files)} file(s) in {DOCS_DIR}:\n")
    total_chunks = 0
    for filename in files:
        print(f"  Processing: {filename} ...")
        result = load_document(filename)
        if result["status"] == "ok":
            print(f"  OK — {result['pages']} pages, {result['chunks']} chunks added")
            total_chunks += result["chunks"]
        else:
            print(f"  ERROR — {result['message']}")

    print(f"\nDone. Total chunks in ChromaDB: {total_chunks}")


def ingest_files(filenames: list):
    total_chunks = 0
    for filename in filenames:
        print(f"Processing: {filename} ...")
        result = load_document(filename)
        if result["status"] == "ok":
            print(f"  OK — {result['pages']} pages, {result['chunks']} chunks added")
            total_chunks += result["chunks"]
        else:
            print(f"  ERROR — {result['message']}")
    print(f"\nDone. Total chunks in ChromaDB: {total_chunks}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ingest_files(sys.argv[1:])
    else:
        ingest_all()
