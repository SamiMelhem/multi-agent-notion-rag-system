# Notion RAG CLI

A comprehensive Retrieval-Augmented Generation (RAG) system for Notion that fetches content, loads it into a vector database, and provides intelligent search with Gemini 2.5 Flash-Lite Preview.

## 🚀 Quick Start Guide

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd notion-rag-cli

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
NOTION_API_KEY=your_notion_api_key_here
NOTION_HOME_PAGE_ID=your_notion_home_page_id_here
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id
```

**Required API Keys:**
- **NOTION_API_KEY**: Get from [Notion Integrations](https://www.notion.so/my-integrations)
- **NOTION_HOME_PAGE_ID**: The Notion page/database ID you want to index (from Notion URL)
- **GEMINI_API_KEY**: Your Google Gemini API key (from Google Cloud Vertex AI)
- **GOOGLE_CLOUD_PROJECT**: Your Google Cloud project ID

### 3. Simple Two-Script Workflow (Recommended)

**First Time Setup:**
```bash
# This will fetch your Notion data and set up everything
python notion_rag_complete.py
```

**Daily Usage:**
```bash
# Quick chat with your existing Notion database
python chat_with_notion.py
```

That's it! The first script sets up everything, and the second script lets you chat quickly.

### 4. Alternative: Interactive CLI Workflow

**Recommended: Use the full interactive workflow:**

```bash
python notion_rag_cli.py workflow -i
```

This single command will:
1. **Fetch**: Let you interactively select which Notion page(s) to fetch
2. **Load**: Automatically process and load content into ChromaDB
3. **Query**: Start an interactive Gemini-powered RAG session

**Alternative: Run each step manually:**

```bash
# Step 1: Fetch Notion content
python notion_rag_cli.py fetch -i

# Step 2: Load into vector database  
python notion_rag_cli.py load -i

# Step 3: Start querying
python notion_rag_cli.py query -i
```

### 5. Ask Questions About Your Notion Database

Once loaded, you can ask questions in natural language:

```bash
# Start interactive query session
python notion_rag_cli.py query -i

# Available commands in interactive mode:
# - help: Show available commands
# - stats: Show collection statistics
# - exit/quit/q: Exit the program

# Special query commands:
# - summarize: [question] - Use summarization template
# - analyze: [question] - Use content analysis template  
# - extract: [question] - Use key points extraction template
# - bullet: [question] - Use bullet-point summary template
```

## 🎯 Simple Workflow Scripts

### `notion_rag_complete.py` - Complete Setup Script
**Use this for first-time setup or when you want fresh data.**

```bash
python notion_rag_complete.py
```

**What it does:**
- 🚀 Fetches all pages from your Notion database
- 📚 Loads content into ChromaDB with embeddings
- 🤖 Initializes Gemini client
- 💬 Starts interactive chat session

**Perfect for:**
- First-time setup
- Fresh data refresh
- Complete workflow in one command

### `chat_with_notion.py` - Quick Chat Script
**Use this for daily conversations with your existing database.**

```bash
python chat_with_notion.py
```

**What it does:**
- ⚡ Quick connection to existing ChromaDB (~4 seconds)
- 🤖 Connects to Gemini API
- 💬 Starts interactive chat immediately
- 💰 Loads cost tracking

**Perfect for:**
- Daily usage
- Quick questions
- When data is already set up

### Script Comparison

| Script | Purpose | Load Time | Use Case | When to Use |
|--------|---------|-----------|----------|-------------|
| `notion_rag_complete.py` | Complete setup + chat | ~15-30 seconds | First time, fresh data | New setup, data refresh |
| `chat_with_notion.py` | Quick chat only | ~4-5 seconds | Daily conversations | Regular usage, existing data |
| `notion_rag_cli.py workflow` | Interactive CLI | ~15-30 seconds | Guided setup | When you want step-by-step control |

## 📁 Core Files

- **`notion_rag_complete.py`** - Complete setup and chat workflow (recommended)
- **`chat_with_notion.py`** - Quick chat with existing data (daily use)
- **`turbo_notion_fetch.py`** - High-performance Notion content fetcher
- **`notion_rag/`** - Core RAG system modules
- **`SYSTEM_DOCUMENTATION.md`** - Detailed system architecture

## 🎯 Features

- ✅ **Simple Two-Script Workflow**: Easy setup and daily usage
- ✅ **Gemini 2.5 Flash-Lite Preview**: Advanced LLM for intelligent responses
- ✅ **Cost Tracking**: Real-time token counting and cost estimation
- ✅ **Prompt Engineering**: Multiple specialized templates (QA, summary, analysis, extraction)
- ✅ **Recursive Fetching**: Automatically fetches main page + all child pages
- ✅ **Vector Database**: ChromaDB with semantic search
- ✅ **Fast Search**: Sub-second query response times
- ✅ **Content Chunking**: Intelligent text segmentation
- ✅ **Metadata Preservation**: Page IDs, URLs, titles, etc.
- ✅ **Performance Monitoring**: Built-in timing and statistics
- ✅ **Secure API Management**: Keyring-based credential storage
- ✅ **Interactive Chat**: Natural language querying with context awareness

## 💰 Cost Optimization

The system includes comprehensive cost tracking:

- **Token Counting**: Accurate input/output token estimation using `tiktoken`
- **Cost Estimation**: Pre-call cost calculation using Gemini pricing
- **Cost Logging**: Persistent cost tracking in `cost_log.json`
- **Cost Summary**: Detailed breakdown of input/output costs

```bash
# View cost summary in interactive mode
> costs

# Check cost log file
cat cost_log.json
```

## 🎨 Prompt Templates

The system includes specialized prompt templates:

- **`rag_qa`**: Standard question answering
- **`rag_summary`**: Detailed summarization
- **`rag_analysis`**: Comprehensive analysis
- **`rag_extraction`**: Key points extraction

```bash
# Use different templates in interactive mode
> summarize: What is cybersecurity?
> analyze: What are the main security threats?
> extract: Key points about incident response
> bullet: List security best practices
```

## 📊 Performance

Typical performance metrics:
- **Fetch Time**: ~0.8s for 9 pages
- **Load Time**: ~14s for 54K characters
- **Search Time**: ~1.4s average per query
- **Processing Rate**: ~4K chars/sec
- **Token Estimation**: Real-time with cost breakdown

## 🔍 Usage Examples

### Simple Two-Script Workflow (Recommended)

```bash
# 1. First time setup - fetches data and starts chatting
python notion_rag_complete.py

# 2. Daily usage - quick chat with existing data
python chat_with_notion.py
```

**Example conversation:**
```
🤔 Your question: What is cybersecurity?
🤖 Answer: Cybersecurity is the practice of protecting networks, devices, people, and data...

🤔 Your question: bullet: List security best practices
🤖 Answer: • Use strong passwords and multi-factor authentication
          • Keep software and systems updated...

🤔 Your question: costs
💰 Cost Summary: Total Cost: $0.003675, Total Entries: 14...
```

### Complete Workflow Example (Alternative)

```bash
# 1. Setup environment and test connections
python notion_rag_cli.py setup

# 2. Run complete workflow
python notion_rag_cli.py workflow -i

# 3. Follow the interactive prompts:
#    - Select pages to fetch (home page, specific page, or search)
#    - Choose recursive fetching if needed
#    - Let the system load content into ChromaDB
#    - Start asking questions!

# Example questions:
> What are the main topics covered in this course?
> summarize: What is the incident response process?
> analyze: What are the key security principles?
> extract: List all the tools mentioned
> bullet: What are the best practices for cybersecurity?
```

### Individual Steps Example

```bash
# Step 1: Fetch content
python notion_rag_cli.py fetch -i
# Choose: 1. Home page only, 2. Home page with children, 3. Specific page, 4. Search

# Step 2: Load into database
python notion_rag_cli.py load -i
# Select collection and process content

# Step 3: Query
python notion_rag_cli.py query -i
# Start asking questions about your content
```

## 🚀 Advanced Features

### Cost Tracking
- Real-time token counting with `tiktoken`
- Pre-call cost estimation
- Persistent cost logging
- Detailed cost breakdowns

### Prompt Engineering
- Multiple specialized templates
- Context-aware prompting
- Chain-of-thought reasoning
- Structured output support

### Vector Search
- Semantic similarity search
- Metadata filtering
- Batch processing
- Collection management

### Notion Integration
- Comprehensive block type support
- Recursive page fetching
- Metadata preservation
- Rich text extraction

## 📝 Environment Variables

Required environment variables:
- `NOTION_API_KEY`: Your Notion API key
- `NOTION_HOME_PAGE_ID`: Your Notion home page ID
- `GEMINI_API_KEY`: Your Gemini API key
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID

## 🔧 Development

### Project Structure
```
notion-rag-cli/
├── notion_rag/           # Core RAG modules
│   ├── config.py        # Configuration management
│   ├── gemini_client.py # Gemini 2.5 Flash-Lite integration
│   ├── vector_store.py  # ChromaDB management
│   ├── embeddings.py    # Embedding generation
│   ├── chunking.py      # Text chunking
│   ├── cost_tracker.py  # Cost tracking
│   ├── prompt_utils.py  # Prompt engineering
│   ├── notion_client.py # Notion API client
│   └── security.py      # Security utilities
├── notion_rag_complete.py    # Complete setup and chat workflow
├── chat_with_notion.py       # Quick chat with existing data
├── turbo_notion_fetch.py     # Notion content fetcher
└── requirements.txt          # Dependencies
```

### Testing
```bash
# Run tests
python -m pytest tests/ -v

# Test specific modules
python -m pytest tests/test_gemini_client.py -v
python -m pytest tests/test_vector_store.py -v
```

## 🆘 Troubleshooting

### Common Issues

1. **"Missing required environment variables"**
   - Ensure all required variables are set in your `.env` file
   - Check that NOTION_API_KEY, NOTION_HOME_PAGE_ID, and GEMINI_API_KEY are set

2. **"No text content found"**
   - The improved text extraction should handle all Notion block types
   - Check that your Notion page has actual content (not just empty blocks)

3. **"Collection not found"**
   - Run `python notion_rag_complete.py` to set up your database first
   - This will create the necessary collections and load your data

4. **"Failed to connect to Gemini API"**
   - Verify your GEMINI_API_KEY is correct
   - Check your internet connection
   - Ensure you have sufficient API quota

### Getting Help

```bash
# Check if your data is loaded
python chat_with_notion.py

# If no data found, set up your database
python notion_rag_complete.py
```

## 📄 License

MIT License - see LICENSE file for details. 