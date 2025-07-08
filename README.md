# Notion RAG CLI

A multi-agent RAG (Retrieval-Augmented Generation) system for Notion API with command-line interface.

## Features

- 🔍 **Vector Database Integration**: ChromaDB for efficient document search and retrieval
- 🔗 **Notion API Integration**: Seamless access to Notion workspaces and pages
- 🛡️ **Secure API Key Management**: Keyring integration for secure credential storage
- 🔄 **Retry Logic**: Robust error handling with configurable retry mechanisms
- 📝 **CLI Interface**: User-friendly command-line interface built with Click
- ✅ **Input Validation**: Pydantic models for robust data validation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd notion_rag_cli
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install in development mode:
```bash
pip install -e .
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your Notion API credentials:
```
NOTION_API_KEY=your_notion_api_key_here
NOTION_DATABASE_ID=your_database_id_here
```

## Usage

### Basic Commands

```bash
# Initialize the RAG system
notion-rag init

# Index Notion pages
notion-rag index --database-id <your-database-id>

# Search documents
notion-rag search "your query here"

# Chat with your documents
notion-rag chat
```

### API Key Management

The CLI uses keyring for secure API key storage. On first run, you'll be prompted to enter your Notion API key.

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
notion_rag_cli/
├── src/notion_rag/          # Main package
│   ├── __init__.py
│   ├── cli.py              # CLI interface
│   ├── models.py           # Pydantic models
│   ├── notion_client.py    # Notion API client
│   ├── vector_store.py     # ChromaDB integration
│   └── utils.py            # Utility functions
├── tests/                   # Test suite
├── docs/                    # Documentation
├── requirements.txt         # Dependencies
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements

- Python 3.8 or higher
- Notion API access token
- ChromaDB for vector storage 