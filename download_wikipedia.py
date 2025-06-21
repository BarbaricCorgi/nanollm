#!/usr/bin/env python3

import os
import urllib.request
import bz2
import json
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar."""
    print(f"Downloading from {url}")
    
    # Get file size
    response = urllib.request.urlopen(url)
    file_size = int(response.headers.get('Content-Length', 0))
    
    # Download with progress bar
    block_size = 8192
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True)
    
    with open(destination, 'wb') as f:
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            progress_bar.update(len(chunk))
    
    progress_bar.close()
    print(f"Downloaded to {destination}")

def check_if_bz2(file_path):
    """Check if file is bz2 compressed."""
    with open(file_path, 'rb') as f:
        magic = f.read(2)
        return magic == b'BZ'

def extract_bz2(file_path, output_path):
    """Extract bz2 compressed file."""
    print(f"Extracting {file_path}...")
    
    with bz2.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            # Read and write in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
    
    print(f"Extracted to {output_path}")

def process_wikipedia_text(input_file, output_file, max_articles=10000):
    """Process Wikipedia text file and extract articles."""
    print(f"Processing Wikipedia articles from {input_file}...")
    
    articles = []
    current_article = []
    article_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading lines"):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_article and article_count < max_articles:
                    # Join the article lines and add to articles
                    article_text = ' '.join(current_article)
                    if len(article_text) > 100:  # Only keep articles with substantial content
                        articles.append(article_text)
                        article_count += 1
                current_article = []
                continue
            
            # Skip special Wikipedia markup
            if line.startswith('</doc>') or line.startswith('<doc'):
                continue
            
            current_article.append(line)
            
            if article_count >= max_articles:
                break
    
    print(f"Extracted {len(articles)} articles")
    
    # Save articles as JSON for easy loading
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved articles to {output_file}")
    return articles

def main():
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # File paths
    url = "https://github.com/GermanT5/wikipedia2corpus/releases/download/v1.0/enwiki-20220201-clean-part-00"
    downloaded_file = "data/enwiki-part-00"
    extracted_file = "data/enwiki-part-00.txt"
    processed_file = "data/wikipedia_articles.json"
    
    # Check if we already have the processed file
    if os.path.exists(processed_file):
        print(f"Processed file already exists: {processed_file}")
        print("Delete it if you want to re-download and process.")
        return
    
    # Download if not exists
    if not os.path.exists(downloaded_file) and not os.path.exists(extracted_file):
        download_file(url, downloaded_file)
    else:
        print(f"Downloaded file already exists")
    
    # Check if file needs extraction
    if not os.path.exists(extracted_file):
        if os.path.exists(downloaded_file):
            # Check if it's compressed
            if check_if_bz2(downloaded_file):
                extract_bz2(downloaded_file, extracted_file)
            else:
                # File is already uncompressed, just rename
                os.rename(downloaded_file, extracted_file)
                print(f"File was already uncompressed, moved to {extracted_file}")
    else:
        print(f"Extracted file already exists: {extracted_file}")
    
    # Process the text file
    process_wikipedia_text(extracted_file, processed_file, max_articles=10000)
    
    # Optionally, clean up intermediate files to save space
    print("\nCleanup options:")
    print(f"1. Keep all files")
    print(f"2. Delete downloaded/extracted text file (save ~1GB)")
    
    choice = input("Enter your choice (1-2, or press Enter to keep all): ").strip()
    
    if choice == '2':
        if os.path.exists(downloaded_file):
            os.remove(downloaded_file)
            print(f"Deleted {downloaded_file}")
        if os.path.exists(extracted_file):
            os.remove(extracted_file)
            print(f"Deleted {extracted_file}")
    
    print("\nDone! Wikipedia articles are ready for training.")

if __name__ == "__main__":
    main()