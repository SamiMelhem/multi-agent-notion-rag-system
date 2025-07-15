# Notion RAG CLI System Documentation

## System Overview

The Notion RAG CLI is a Retrieval-Augmented Generation (RAG) system designed to interact with Notion workspaces through a streamlined two-script workflow. The system provides secure API key management, robust error handling, and efficient document processing capabilities with Google Gemini 2.5 Flash-Lite Preview integration.

## Current System State

### Version: 0.1.0
### Author: Sami Melhem
### Email: samilmelhem23@gmail.com

## Architecture

### Core Components

1. **Complete RAG System** (`notion_rag_complete.py`)
   - Main workflow script combining fetch, load, and chat
   - Interactive page selection and querying
   - Cost tracking and prompt engineering
   - Version: 0.1.0

2. **Quick Chat Interface** (`chat_with_notion.py`)
   - Lightweight script for daily chat with existing database
   - Fast initialization for quick queries
   - Multiple prompt templates and cost tracking

3. **Configuration Management** (`notion_rag/config.py`)
   - Uses Pydantic for data validation
   - Supports environment variables and .env files
   - Integrates with secure keyring storage
   - ChromaDB configuration with nested environment variables

4. **Security Layer** (`notion_rag/security.py`)
   - Secure API key management using system keyring
   - Input validation and sanitization
   - Pydantic models for secure data handling

5. **Notion API Client** (`notion_rag/notion_client.py`)
   - Async-capable client with rate limiting
   - Comprehensive error handling
   - Support for pages, databases, and blocks

6. **Gemini Integration** (`notion_rag/gemini_client.py`)
   - Google Gemini 2.5 Flash-Lite Preview integration
   - Function calling and structured output support
   - RAG completion with context documents

7. **Vector Store** (`notion_rag/vector_store.py`)
   - ChromaDB integration for document storage
   - BAAI/bge-small-en-v1.5 embeddings
   - Document chunking and similarity search

## Environment Variables

The system uses the following environment variables:

### Required Environment Variables
- `NOTION_API_KEY`: Your Notion API integration token
- `NOTION_HOME_PAGE_ID`: The ID of your Notion home page
- `GEMINI_API_KEY`: Google Gemini API key
- `GOOGLE_CLOUD_PROJECT`: Google Cloud Project ID for Vertex AI

## Dependencies

### Core Dependencies
- `chromadb==1.0.15`: Vector database for document storage and similarity search
- `google-generativeai==0.8.3`: Google Gemini 2.5 Flash-Lite Preview API client
- `sentence-transformers==2.5.1`: BAAI/bge-small-en-v1.5 embedding generation
- `tiktoken==0.7.0`: Token counting and cost tracking for Gemini API
- `notion-client==2.4.0`: Official Notion API client for content fetching
- `python-dotenv==1.1.1`: Environment variable management from .env files
- `pydantic==2.11.7`: Data validation and settings management
- `pydantic-settings==2.10.1`: Advanced settings management with environment variables

### Required by Dependencies
- `requests>=2.32.0`: HTTP client for API communications
- `numpy>=2.3.0`: Numerical computing for embeddings and vector operations
- `typing-extensions>=4.14.0`: Type hints for Python compatibility
- `urllib3>=2.5.0`: HTTP library for secure connections

## Current Features

### ✅ Implemented
1. **Secure API Key Management**
   - System keyring integration
   - Automatic key retrieval and storage
   - Support for multiple API keys (Notion, Gemini, OpenAI, HuggingFace)

2. **Configuration System**
   - Environment-based configuration with nested variables
   - Pydantic validation
   - ChromaDB configuration with double underscore syntax

3. **Two-Script Workflow**
   - `notion_rag_complete.py`: Complete setup and interactive chat
   - `chat_with_notion.py`: Quick daily chat with existing database

4. **Notion API Integration**
   - Page content retrieval with `turbo_notion_fetch.py`
   - Database querying
   - Block children fetching
   - Rate limiting and error handling

5. **Gemini 2.5 Flash-Lite Preview Integration**
   - Advanced function calling capabilities
   - Structured output support
   - RAG completion with context documents
   - Multiple prompt templates

6. **ChromaDB Vector Database Integration**
   - Local persistent storage
   - Collection management utilities
   - Document chunking and storage (512-token chunks, 50-token overlap)
   - Vector similarity search with BAAI/bge-small-en-v1.5 embeddings
   - Batch processing support

7. **Cost Tracking and Optimization**
   - Token counting with tiktoken
   - Cost estimation for Gemini API calls
   - Usage statistics and summaries

8. **Prompt Engineering**
   - Multiple RAG prompt templates (rag_qa, rag_summary, rag_analysis, rag_extraction)
   - Special query prefixes (summarize:, analyze:, extract:)
   - Customizable prompt library with task-specific templates
   - Context-aware responses with source citation

9. **Security Features**
   - Input sanitization
   - API key validation
   - Notion ID validation

## File Structure

```
notion-rag-cli/
├── notion_rag/
│   ├── __init__.py          # Package initialization
│   ├── config.py           # Configuration management
│   ├── security.py         # Security utilities
│   ├── notion_client.py    # Notion API client
│   ├── gemini_client.py    # Gemini API client
│   ├── vector_store.py     # ChromaDB integration
│   ├── embeddings.py       # Embedding generation
│   ├── chunking.py         # Text chunking utilities
│   ├── cost_tracker.py     # Cost tracking and optimization
│   └── prompt_utils.py     # Prompt engineering utilities
├── tests/
│   ├── test_config.py
│   ├── test_notion_client.py
│   ├── test_security.py
│   └── test_vector_store.py
├── notion_rag_complete.py  # Complete workflow script
├── chat_with_notion.py     # Quick chat script
├── turbo_notion_fetch.py   # Notion content fetcher
├── requirements.txt        # Dependencies
├── README.md              # User documentation
└── SYSTEM_DOCUMENTATION.md # This file
```

## Usage Examples

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NOTION_API_KEY="your_api_key_here"
export NOTION_HOME_PAGE_ID="your_home_page_id_here"

# Complete workflow (fetch, load, chat)
python notion_rag_complete.py

# Quick chat with existing database
python chat_with_notion.py
```

### Environment Setup
```bash
# Create .env file
cat > .env << EOF
NOTION_API_KEY=your_notion_api_key_here
NOTION_HOME_PAGE_ID=your_home_page_id_here
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id_here
EOF
```

### Available Commands in Interactive Mode

#### notion_rag_complete.py Commands
- `fetch`: Fetch pages from Notion
- `load`: Load cached pages to vector database
- `query`: Search and chat with documents
- `workflow`: Complete fetch → load → query workflow
- `stats`: Show database statistics
- `costs`: Show cost summary
- `templates`: Show available prompt templates
- `help`: Show help information
- `quit`: Exit the application

#### chat_with_notion.py Commands
- `help`: Show available commands
- `stats`: Show database statistics
- `costs`: Show cost summary
- `templates`: Show available prompt templates
- `quit`/`exit`: Exit the chat

#### Special Query Prefixes
- `summarize:`: Use RAG summarization template for detailed summaries with context
- `analyze:`: Use RAG analysis template for comprehensive content analysis
- `extract:`: Use RAG extraction template for key points and facts extraction

## Security Considerations

1. **API Key Storage**: Uses system keyring for secure storage
2. **Input Validation**: All inputs are validated and sanitized
3. **Error Handling**: Comprehensive error handling without exposing sensitive data
4. **Rate Limiting**: Built-in rate limiting for API calls
5. **Environment Variables**: Secure handling of nested configuration

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

## Cost Optimization

### Token Counting
- Uses tiktoken for accurate token counting
- Tracks input and output tokens separately
- Provides cost estimates for Gemini API calls

### Embedding Optimization
- Uses efficient BAAI/bge-small-en-v1.5 model
- 512-token chunks with 50-token overlap
- Batch processing for large datasets

### Prompt Engineering
- Multiple RAG templates (rag_qa, rag_summary, rag_analysis, rag_extraction)
- Special query prefixes (summarize:, analyze:, extract:)
- Context-aware responses with source citation
- Structured output for better parsing

## Next Steps

### Immediate Priorities
1. Enhanced error recovery and retry logic
2. Incremental database updates
3. Export/import functionality
4. Web interface development

### Future Enhancements
1. Support for multiple Notion workspaces
2. Document versioning and change tracking
3. Advanced analytics and insights
4. Integration with other LLM providers

## Troubleshooting

### Common Issues
1. **API Key Not Found**: Ensure environment variables are set or use keyring storage
2. **Gemini API Errors**: Verify GEMINI_API_KEY and Google Cloud Project setup
3. **Rate Limiting**: System automatically handles rate limits with retry logic

### Debug Mode
Enable debug logging by setting logging level in your Python environment.

## Support

For issues and questions:
- Email: samilmelhem23@gmail.com
- Check the README.md for detailed usage instructions
- Review test files for implementation examples

---

**Last Updated**: Current session
**System Status**: Production ready with two-script workflow
**Version**: 0.1.0 