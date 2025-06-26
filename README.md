# 🏥 Medical Evidence RAG System
A smart medical research assistant that compares US and Dutch diabetes treatment guidelines using AI.

## What This Does
This app helps medical researchers quickly find and compare treatment guidelines from different countries. It uses two different AI search methods to give you better answers about medical questions.

## 🚀 **[Try the Live Demo](https://med-rag1.streamlit.app/)**
*Click above to explore the interactive application - no setup required!*
![Medical RAG System Interface](screenshots/main-interface.png)


## Key Features
🚀 Two Search Methods
- **Fast Vector Search**: Quick similarity matching (like Google search)

- **Knowledge Graph Search**: Smarter search that understands relationships between medical concepts

## 📊 Real-Time Comparison
- See which search method gives better answers

- Compare response times and quality scores

- Get detailed performance metrics

## 🕸️ Interactive Knowledge Graph
- Visual map showing how medical concepts connect

- Color-coded by category (treatments, conditions, guidelines)

- Click and explore relationships between diabetes treatments

## 📚 Document Database
- Browse all source documents (US CDC and Dutch RIVM guidelines)

- Search across all documents

- See document statistics and content analysis

## Sample Questions You Can Ask
"Compare US and Dutch approaches to Type 2 diabetes management"

"What are the medication protocols for diabetes in different countries?"

"How do dietary recommendations differ between US and Netherlands?"

"Compare weight loss targets in diabetes prevention programs"

## Quality Metrics Explained
The app measures answer quality using:

- Source Diversity: How well it covers both US and Dutch sources

- Clinical Coverage: How many medical concepts it includes

- Synthesis Quality: How well it compares different approaches

- Evidence Strength: How confident the AI is in its sources

## Technical Stack
AI Model: Google Gemma-3-4B (runs locally via LM Studio)

Search Engine: LlamaIndex with ColBERT reranking

Knowledge Graphs: NetworkX + PyVis for interactive visualization

Interface: Streamlit web app

## Demo Mode
This app runs in demo mode with pre-generated responses, so you can explore all features without needing to set up the AI model locally.

*Built to showcase advanced RAG (Retrieval-Augmented Generation) techniques for medical research applications.*