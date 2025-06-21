#!/usr/bin/env python3

import os
import json
import urllib.request
from tqdm import tqdm

def download_text_samples():
    """Download sample text data from various sources."""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Sample text sources (using public domain texts)
    sources = [
        {
            "name": "Shakespeare Sonnets",
            "url": "https://www.gutenberg.org/files/1041/1041-0.txt",
            "description": "Complete sonnets of William Shakespeare"
        },
        {
            "name": "Alice in Wonderland",
            "url": "https://www.gutenberg.org/files/11/11-0.txt",
            "description": "Alice's Adventures in Wonderland by Lewis Carroll"
        },
        {
            "name": "Pride and Prejudice",
            "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
            "description": "Pride and Prejudice by Jane Austen"
        }
    ]
    
    texts = []
    
    print("Downloading sample texts from Project Gutenberg...")
    
    for source in sources:
        print(f"\nDownloading: {source['name']}")
        print(f"Description: {source['description']}")
        
        try:
            with urllib.request.urlopen(source['url']) as response:
                content = response.read().decode('utf-8')
                
            # Clean up the text
            lines = content.split('\n')
            
            # Find start and end markers for Project Gutenberg texts
            start_idx = 0
            end_idx = len(lines)
            
            for i, line in enumerate(lines):
                if "START OF THE PROJECT GUTENBERG" in line or "START OF THIS PROJECT GUTENBERG" in line:
                    start_idx = i + 1
                    break
            
            for i in range(len(lines) - 1, 0, -1):
                if "END OF THE PROJECT GUTENBERG" in line or "END OF THIS PROJECT GUTENBERG" in lines[i]:
                    end_idx = i
                    break
            
            # Extract the actual content
            book_lines = lines[start_idx:end_idx]
            
            # Split into paragraphs
            current_paragraph = []
            for line in book_lines:
                line = line.strip()
                if line:
                    current_paragraph.append(line)
                elif current_paragraph:
                    paragraph = ' '.join(current_paragraph)
                    if len(paragraph) > 100:  # Only keep substantial paragraphs
                        texts.append(paragraph)
                    current_paragraph = []
            
            print(f"Extracted {len([t for t in texts if source['name'] in t or len(texts) > 0])} paragraphs")
            
        except Exception as e:
            print(f"Error downloading {source['name']}: {e}")
    
    print(f"\nTotal paragraphs collected: {len(texts)}")
    
    # Save the texts
    output_file = "data/literature_texts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    
    print(f"Saved texts to {output_file}")
    
    # Create a sample of Wikipedia-style texts for variety
    wiki_style_texts = [
        "Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.",
        
        "The Python programming language is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together.",
        
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The result is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them.",
        
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs.",
        
        "The transformer is a deep learning model introduced in 2017, used primarily in the field of natural language processing. Like recurrent neural networks, transformers are designed to handle sequential input data, such as natural language, for tasks such as translation and text summarization. However, unlike RNNs, transformers do not necessarily process the data in order. Rather, the attention mechanism provides context for any position in the input sequence.",
    ] * 20  # Repeat to have more data
    
    # Combine all texts
    all_texts = texts + wiki_style_texts
    
    # Save combined dataset
    combined_file = "data/combined_texts.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    
    print(f"\nCreated combined dataset with {len(all_texts)} texts")
    print(f"Saved to {combined_file}")
    
    return all_texts

def main():
    print("Downloading sample training data...")
    print("=" * 50)
    
    texts = download_text_samples()
    
    print("\nData download complete!")
    print(f"Total texts available for training: {len(texts)}")
    print("\nYou can now run training with:")
    print("  python3 train_on_samples.py")

if __name__ == "__main__":
    main()