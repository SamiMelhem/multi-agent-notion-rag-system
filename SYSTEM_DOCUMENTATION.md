# Notion RAG CLI System Documentation

## System Overview

The Notion RAG CLI is a multi-agent RAG (Retrieval-Augmented Generation) system designed to interact with Notion workspaces through a command-line interface. The system provides secure API key management, robust error handling, and efficient document processing capabilities.

## Current System State

### Version: 0.1.0
### Author: Sami Melhem
### Email: samilmelhem23@gmail.com

## Architecture

### Core Components

1. **CLI Interface** (`notion_rag/cli.py`)
   - Built with Click framework
   - Provides commands: `init`, `index`, `search`, `chat`
   - Version: 0.1.0

2. **Configuration Management** (`notion_rag/config.py`)
   - Uses Pydantic for data validation
   - Supports environment variables and .env files
   - Integrates with secure keyring storage

3. **Security Layer** (`notion_rag/security.py`)
   - Secure API key management using system keyring
   - Input validation and sanitization
   - Pydantic models for secure data handling

4. **Notion API Client** (`notion_rag/notion_client.py`)
   - Async-capable client with rate limiting
   - Comprehensive error handling
   - Support for pages, databases, and blocks

## Environment Variables

The system uses the following environment variables:
- `NOTION_API_KEY`: Your Notion API integration token
- `NOTION_HOME_PAGE_ID`: The ID of your Notion home page
- `NOTION_DATABASE_ID`: (Optional) Default database ID for operations

## Dependencies

### Core Dependencies
- `notion-client==2.4.0`: Official Notion API client
- `chromadb==1.0.15`: Vector database for document storage
- `click==8.2.1`: CLI framework
- `pydantic==2.11.7`: Data validation
- `keyring==25.6.0`: Secure credential storage
- `python-dotenv==1.1.1`: Environment variable management

### Development Dependencies
- `pytest==8.4.1`: Testing framework
- `black==25.1.0`: Code formatting
- `mypy==1.16.1`: Type checking
- `flake8==7.3.0`: Linting

## Current Features

### ✅ Implemented
1. **Secure API Key Management**
   - System keyring integration
   - Automatic key retrieval and storage
   - Support for multiple API keys (Notion, OpenAI, HuggingFace)

2. **Configuration System**
   - Environment-based configuration
   - Pydantic validation
   - ChromaDB configuration

3. **CLI Framework**
   - Basic command structure
   - Interactive chat mode (skeleton)
   - Search and indexing commands (skeleton)
   - Collection management commands

4. **Notion API Integration**
   - Page content retrieval
   - Database querying
   - Block children fetching
   - Rate limiting and error handling

5. **Security Features**
   - Input sanitization
   - API key validation
   - Notion ID validation

6. **ChromaDB Vector Database Integration**
   - Local persistent storage
   - Collection management utilities
   - Document chunking and storage
   - Vector search capabilities
   - Batch processing support
   - Text chunking with 512-token chunks and 50-token overlap
   - Embedding generation using BAAI/bge-small-en-v1.5
   - Vector similarity search with automatic embedding generation

### 🚧 In Progress / TODO
1. **Notion Content Processing**
   - Convert Notion pages to text chunks
   - Extract structured data from Notion blocks
   - Handle different Notion content types

2. **RAG Pipeline**
   - Document retrieval and ranking
   - Response generation with LLMs
   - Context management

3. **Chat Interface**
   - Interactive conversation with documents
   - Conversation history
   - Response streaming

4. **Indexing System**
   - Batch processing of Notion pages
   - Incremental updates
   - Progress tracking

## File Structure

```
notion-rag-cli/
├── notion_rag/
│   ├── __init__.py          # Package initialization
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── security.py         # Security utilities
│   ├── notion_client.py    # Notion API client
│   └── vector_store.py     # ChromaDB integration
├── tests/
│   ├── test_config.py
│   ├── test_notion_client.py
│   ├── test_security.py
│   └── test_vector_store.py
├── requirements.txt        # Dependencies
├── README.md              # User documentation
└── SYSTEM_DOCUMENTATION.md # This file
```

## Usage Examples

### Basic Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NOTION_API_KEY="your_api_key_here"
export NOTION_HOME_PAGE_ID="your_home_page_id_here"

# Run CLI
python -m notion_rag.cli --help
```

### Available Commands
```bash
# Initialize system
notion-rag init --database-id <database_id>

# Collection management
notion-rag collections                    # List all collections
notion-rag collection-info -c <name>     # Get collection details
notion-rag clear-collection -c <name>    # Clear collection
notion-rag delete-collection -c <name>   # Delete collection

# Text processing
notion-rag chunk -t "your text here"     # Chunk text into segments
notion-rag embed -t "your text here"     # Generate embeddings
notion-rag add-text -c <collection> -t "text" # Add text with chunking

# Index documents
notion-rag index --database-id <database_id> --batch-size 100

# Search documents
notion-rag search "your query here" --limit 10

# Interactive chat
notion-rag chat
```

## Security Considerations

1. **API Key Storage**: Uses system keyring for secure storage
2. **Input Validation**: All inputs are validated and sanitized
3. **Error Handling**: Comprehensive error handling without exposing sensitive data
4. **Rate Limiting**: Built-in rate limiting for API calls

## Development Guidelines

### Code Style
- Use Black for code formatting
- Follow PEP 8 guidelines
- Use type hints throughout
- Write comprehensive docstrings

### Testing
- Use pytest for testing
- Maintain good test coverage
- Test security features thoroughly

### Error Handling
- Use custom exception classes
- Provide meaningful error messages
- Log errors appropriately

## Next Steps

### Immediate Priorities
1. Add document embedding functionality
2. Complete the indexing system
3. Implement vector search
4. Enhance RAG pipeline

### Future Enhancements
1. Add support for multiple Notion workspaces
2. Implement document versioning
3. Add export/import functionality
4. Create web interface

## Troubleshooting

### Common Issues
1. **API Key Not Found**: Ensure NOTION_API_KEY is set in environment or keyring
2. **Rate Limiting**: System automatically handles rate limits with retry logic
3. **Invalid Page ID**: Use the InputValidator to validate Notion IDs

### Debug Mode
Enable debug logging by setting log level to DEBUG in configuration.

## Support

For issues and questions:
- Email: samilmelhem23@gmail.com
- Check the README.md for detailed usage instructions
- Review test files for implementation examples

---

**Last Updated**: Current session
**System Status**: Development in progress
**Version**: 0.1.0 