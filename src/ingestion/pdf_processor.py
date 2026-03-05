"""
Improved PDF Processor for 45Labs
Better chunking strategy that creates more consistent chunks.
"""

import os
import json
from typing import List, Dict, Any
import pdfplumber
from pathlib import Path


class PDFProcessor:
    """Process PDF documents and extract text chunks with metadata."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Target size of each chunk in words
            overlap: Number of overlapping words between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_content.append(f"[Page {page_num}]\n{text}")
        
        return "\n\n".join(text_content)
    
    def chunk_text_sliding_window(self, text: str, source: str, component: str = "Extended Essay") -> List[Dict[str, Any]]:
        """
        Split text into chunks using sliding window approach.
        This ensures more consistent chunk sizes.
        
        Args:
            text: Full text to chunk
            source: Source document name
            component: Component type
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # Split into words
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            return []
        
        chunks = []
        chunk_index = 0
        start_idx = 0
        
        while start_idx < total_words:
            # Get chunk of words
            end_idx = min(start_idx + self.chunk_size, total_words)
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {
                    "component": component,
                    "source": source,
                    "chunk_index": chunk_index,
                    "word_count": len(chunk_words),
                    "start_word": start_idx,
                    "end_word": end_idx
                }
            })
            
            chunk_index += 1
            
            # Move window forward with overlap
            start_idx += self.chunk_size - self.overlap
        
        return chunks
    
    def chunk_text_semantic(self, text: str, source: str, component: str = "Extended Essay") -> List[Dict[str, Any]]:
        """
        Original semantic chunking (splits by paragraphs).
        """
        chunks = []
        sections = text.split('\n\n')
        current_chunk = ""
        chunk_words = 0
        chunk_index = 0
        
        for section in sections:
            section_words = self.count_words(section)
            
            if section_words > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {
                            "component": component,
                            "source": source,
                            "chunk_index": chunk_index,
                            "word_count": chunk_words
                        }
                    })
                    chunk_index += 1
                
                words = section.split()
                current_chunk = ""
                chunk_words = 0
                
                for word in words:
                    if chunk_words >= self.chunk_size and current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": {
                                "component": component,
                                "source": source,
                                "chunk_index": chunk_index,
                                "word_count": chunk_words
                            }
                        })
                        chunk_index += 1
                        
                        overlap_words = current_chunk.split()[-self.overlap:]
                        current_chunk = " ".join(overlap_words) + " " + word
                        chunk_words = len(current_chunk.split())
                    else:
                        current_chunk += (" " + word) if current_chunk else word
                        chunk_words = len(current_chunk.split())
            else:
                if chunk_words + section_words > self.chunk_size and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {
                            "component": component,
                            "source": source,
                            "chunk_index": chunk_index,
                            "word_count": chunk_words
                        }
                    })
                    chunk_index += 1
                    current_chunk = section
                    chunk_words = section_words
                else:
                    current_chunk += "\n\n" + section if current_chunk else section
                    chunk_words += section_words
        
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {
                    "component": component,
                    "source": source,
                    "chunk_index": chunk_index,
                    "word_count": chunk_words
                }
            })
        
        return chunks
    
    def process_pdf(self, pdf_path: str, component: str = "Extended Essay", 
                   strategy: str = "sliding") -> List[Dict[str, Any]]:
        """
        Process a PDF file and return chunks with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            component: Component type
            strategy: "sliding" for consistent chunks or "semantic" for paragraph-based
            
        Returns:
            List of chunk dictionaries
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            raise ValueError(f"No text extracted from {pdf_path}")
        
        print(f"Extracted {len(text)} characters")
        print(f"Total words: {self.count_words(text)}")
        
        # Get source name from file path
        source = Path(pdf_path).stem
        
        # Chunk the text with selected strategy
        if strategy == "sliding":
            chunks = self.chunk_text_sliding_window(text, source, component)
            print(f"Created {len(chunks)} chunks using sliding window")
        else:
            chunks = self.chunk_text_semantic(text, source, component)
            print(f"Created {len(chunks)} chunks using semantic splitting")
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str):
        """Save chunks to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"Chunks saved to: {output_path}")


if __name__ == "__main__":
    # Test both strategies
    pdf_path = "data/rubrics/ib_extended_essay_guide.pdf"
    
    # Sliding window (more consistent chunks)
    print("\n" + "="*60)
    print("SLIDING WINDOW STRATEGY (chunk_size=300)")
    print("="*60)
    processor = PDFProcessor(chunk_size=300, overlap=50)
    chunks_sliding = processor.process_pdf(pdf_path, strategy="sliding")
    processor.save_chunks(chunks_sliding, "data/rubrics/chunks_sliding.json")
    
    # Semantic (paragraph-based)
    print("\n" + "="*60)
    print("SEMANTIC STRATEGY (chunk_size=300)")
    print("="*60)
    processor = PDFProcessor(chunk_size=300, overlap=50)
    chunks_semantic = processor.process_pdf(pdf_path, strategy="semantic")
    processor.save_chunks(chunks_semantic, "data/rubrics/chunks_semantic.json")
    
    print(f"\nComparison:")
    print(f"  Sliding window: {len(chunks_sliding)} chunks")
    print(f"  Semantic: {len(chunks_semantic)} chunks")
