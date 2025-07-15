"""
Command-line interface for the Notion RAG system.
"""

import click
import sys
import os
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional
from .config import Config
from .vector_store import ChromaDBManager, DocumentChunk
from .embeddings import create_embedding_generator
from .chunking import chunk_text, count_tokens
from .gemini_client import create_gemini_client, GeminiMessage
from .notion_client import NotionClient


def extract_text_from_notion_blocks(blocks: List[Dict[str, Any]]) -> str:
    """
    Extract text content from Notion blocks with comprehensive block type support.
    
    Args:
        blocks: List of Notion block objects
        
    Returns:
        str: Extracted text content
    """
    text_content = []
    
    for block in blocks:
        block_type = block.get("type", "")
        
        # Handle different block types
        if block_type == "paragraph":
            rich_text = block.get("paragraph", {}).get("rich_text", [])
            if rich_text:
                text_content.append(rich_text[0].get("plain_text", ""))
                
        elif block_type == "heading_1":
            rich_text = block.get("heading_1", {}).get("rich_text", [])
            if rich_text:
                text_content.append(f"# {rich_text[0].get('plain_text', '')}")
                
        elif block_type == "heading_2":
            rich_text = block.get("heading_2", {}).get("rich_text", [])
            if rich_text:
                text_content.append(f"## {rich_text[0].get('plain_text', '')}")
                
        elif block_type == "heading_3":
            rich_text = block.get("heading_3", {}).get("rich_text", [])
            if rich_text:
                text_content.append(f"### {rich_text[0].get('plain_text', '')}")
                
        elif block_type == "bulleted_list_item":
            rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
            if rich_text:
                text_content.append(f"• {rich_text[0].get('plain_text', '')}")
                
        elif block_type == "numbered_list_item":
            rich_text = block.get("numbered_list_item", {}).get("rich_text", [])
            if rich_text:
                text_content.append(f"1. {rich_text[0].get('plain_text', '')}")
                
        elif block_type == "to_do":
            rich_text = block.get("to_do", {}).get("rich_text", [])
            checked = block.get("to_do", {}).get("checked", False)
            if rich_text:
                checkbox = "[x]" if checked else "[ ]"
                text_content.append(f"{checkbox} {rich_text[0].get('plain_text', '')}")
                
        elif block_type == "toggle":
            rich_text = block.get("toggle", {}).get("rich_text", [])
            if rich_text:
                text_content.append(f"▶ {rich_text[0].get('plain_text', '')}")
                
        elif block_type == "quote":
            rich_text = block.get("quote", {}).get("rich_text", [])
            if rich_text:
                text_content.append(f"> {rich_text[0].get('plain_text', '')}")
                
        elif block_type == "callout":
            rich_text = block.get("callout", {}).get("rich_text", [])
            if rich_text:
                icon = block.get("callout", {}).get("icon", {}).get("emoji", "💡")
                text_content.append(f"{icon} {rich_text[0].get('plain_text', '')}")
                
        elif block_type == "code":
            rich_text = block.get("code", {}).get("rich_text", [])
            language = block.get("code", {}).get("language", "")
            if rich_text:
                text_content.append(f"```{language}")
                text_content.append(rich_text[0].get("plain_text", ""))
                text_content.append("```")
                
        elif block_type == "divider":
            text_content.append("---")
            
        elif block_type == "table_of_contents":
            text_content.append("[Table of Contents]")
            
        elif block_type == "breadcrumb":
            text_content.append("[Breadcrumb Navigation]")
            
        elif block_type == "column_list":
            # Handle column content
            children = block.get("column_list", {}).get("children", [])
            for child in children:
                child_text = extract_text_from_notion_blocks(child)
                if child_text:
                    text_content.append(child_text)
                    
        elif block_type == "synced_block":
            # Handle synced block content
            children = block.get("synced_block", {}).get("children", [])
            for child in children:
                child_text = extract_text_from_notion_blocks(child)
                if child_text:
                    text_content.append(child_text)
                    
        elif block_type == "template":
            # Handle template content
            children = block.get("template", {}).get("children", [])
            for child in children:
                child_text = extract_text_from_notion_blocks(child)
                if child_text:
                    text_content.append(child_text)
                    
        elif block_type == "link_to_page":
            # Handle link to page
            page_id = block.get("link_to_page", {}).get("page_id", "")
            text_content.append(f"[Link to page: {page_id}]")
            
        elif block_type == "child_page":
            # Handle child page
            title = block.get("child_page", {}).get("title", "Untitled")
            text_content.append(f"[Child Page: {title}]")
            
        elif block_type == "child_database":
            # Handle child database
            title = block.get("child_database", {}).get("title", "Untitled")
            text_content.append(f"[Child Database: {title}]")
            
        elif block_type == "embed":
            # Handle embed
            url = block.get("embed", {}).get("url", "")
            text_content.append(f"[Embed: {url}]")
            
        elif block_type == "image":
            # Handle image
            url = block.get("image", {}).get("external", {}).get("url", "")
            if not url:
                url = block.get("image", {}).get("file", {}).get("url", "")
            text_content.append(f"[Image: {url}]")
            
        elif block_type == "video":
            # Handle video
            url = block.get("video", {}).get("external", {}).get("url", "")
            if not url:
                url = block.get("video", {}).get("file", {}).get("url", "")
            text_content.append(f"[Video: {url}]")
            
        elif block_type == "file":
            # Handle file
            url = block.get("file", {}).get("external", {}).get("url", "")
            if not url:
                url = block.get("file", {}).get("file", {}).get("url", "")
            text_content.append(f"[File: {url}]")
            
        elif block_type == "pdf":
            # Handle PDF
            url = block.get("pdf", {}).get("external", {}).get("url", "")
            if not url:
                url = block.get("pdf", {}).get("file", {}).get("url", "")
            text_content.append(f"[PDF: {url}]")
            
        elif block_type == "bookmark":
            # Handle bookmark
            url = block.get("bookmark", {}).get("url", "")
            text_content.append(f"[Bookmark: {url}]")
            
        elif block_type == "equation":
            # Handle equation
            expression = block.get("equation", {}).get("expression", "")
            text_content.append(f"[Equation: {expression}]")
            
        elif block_type == "table":
            # Handle table
            text_content.append("[Table Content]")
            
        elif block_type == "table_row":
            # Handle table row
            cells = block.get("table_row", {}).get("cells", [])
            row_text = " | ".join([cell[0].get("plain_text", "") if cell else "" for cell in cells])
            text_content.append(row_text)
            
        else:
            # For unknown block types, try to extract any rich_text
            if "rich_text" in block:
                rich_text = block["rich_text"]
                if rich_text:
                    text_content.append(rich_text[0].get("plain_text", ""))
    
    return "\n\n".join(text_content)


def check_environment() -> bool:
    """Check if required environment variables are set."""
    required_vars = ["NOTION_API_KEY", "NOTION_HOME_PAGE_ID"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        click.echo("❌ Missing required environment variables:")
        for var in missing_vars:
            click.echo(f"   - {var}")
        click.echo("\n💡 Please set these variables in your .env file or environment.")
        return False
    
    return True


def get_notion_client() -> Optional[NotionClient]:
    """Get Notion client with error handling."""
    try:
        return NotionClient()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {str(e)}")
        return None
    except Exception as e:
        click.echo(f"❌ Failed to initialize Notion client: {str(e)}")
        return None


@click.group()
@click.version_option(version="0.1.0")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    Notion RAG CLI - A comprehensive RAG system for Notion API.
    
    Features:
    - Interactive page selection and fetching
    - Vector database management
    - RAG queries with Gemini 2.5 Flash-Lite Preview
    - Cost tracking and optimization
    - Prompt engineering templates
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config()


@cli.command()
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactive mode for page selection",
)
@click.option(
    "--page-id",
    "-p",
    help="Specific Notion page ID to fetch",
    type=str,
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Fetch all child pages recursively",
)
@click.option(
    "--output-dir",
    "-o",
    help="Output directory for fetched content",
    default="./notion_content",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.pass_context
def fetch(
    ctx: click.Context,
    interactive: bool,
    page_id: Optional[str],
    recursive: bool,
    output_dir: str,
) -> None:
    """Fetch content from Notion pages."""
    if not check_environment():
        raise click.Abort()
    
    config = ctx.obj["config"]
    
    try:
        notion_client = get_notion_client()
        if not notion_client:
            raise click.Abort()
        
        if interactive:
            # Interactive page selection
            click.echo("🔍 Interactive Notion Page Selection")
            click.echo("=" * 50)
            
            # Get home page info
            home_page_id = os.getenv("NOTION_HOME_PAGE_ID")
            click.echo(f"🏠 Home page ID: {home_page_id}")
            
            # Show available options
            click.echo("\n📋 Available options:")
            click.echo("1. Fetch home page only")
            click.echo("2. Fetch home page with all children")
            click.echo("3. Enter specific page ID")
            click.echo("4. Search for pages")
            
            choice = click.prompt("\nSelect option", type=click.Choice(["1", "2", "3", "4"]))
            
            if choice == "1":
                page_id = home_page_id
                recursive = False
            elif choice == "2":
                page_id = home_page_id
                recursive = True
            elif choice == "3":
                page_id = click.prompt("Enter page ID")
                recursive = click.confirm("Fetch child pages recursively?")
            elif choice == "4":
                query = click.prompt("Enter search query")
                pages = notion_client.search_pages(query)
                if pages:
                    click.echo(f"\n📄 Found {len(pages)} pages:")
                    for i, page in enumerate(pages[:10], 1):  # Show first 10
                        title = page.get("properties", {}).get("title", {}).get("title", [{}])[0].get("plain_text", "Untitled")
                        click.echo(f"{i}. {title} (ID: {page['id']})")
                    
                    if len(pages) > 10:
                        click.echo(f"... and {len(pages) - 10} more pages")
                    
                    page_choice = click.prompt("Select page number", type=int, default=1)
                    if 1 <= page_choice <= len(pages):
                        page_id = pages[page_choice - 1]["id"]
                        recursive = click.confirm("Fetch child pages recursively?")
                    else:
                        click.echo("❌ Invalid selection")
                        return
                else:
                    click.echo("❌ No pages found")
                    return
        
        if not page_id:
            click.echo("❌ No page ID specified")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"\n🚀 Fetching page: {page_id}")
        if recursive:
            click.echo("📂 Including all child pages")
        
        # Fetch page content
        page_content = notion_client.get_page_content(page_id)
        
        # Extract page title
        title = "Untitled"
        if "properties" in page_content:
            title_prop = page_content["properties"].get("title", {})
            if "title" in title_prop:
                title = title_prop["title"][0]["plain_text"]
        
        click.echo(f"📄 Page title: {title}")
        
        # Get page blocks
        blocks = notion_client.get_block_children(page_id)
        click.echo(f"📝 Found {len(blocks)} blocks")
        
        # Save content
        content_file = output_path / f"{page_id}.json"
        import json
        with open(content_file, 'w', encoding='utf-8') as f:
            json.dump({
                "page_id": page_id,
                "title": title,
                "blocks": blocks,
                "recursive": recursive
            }, f, indent=2, ensure_ascii=False)
        
        click.echo(f"✅ Content saved to: {content_file}")
        
        if recursive:
            # Get child pages
            child_pages = notion_client.get_all_child_pages(page_id)
            click.echo(f"📂 Found {len(child_pages)} child pages")
            
            for child in child_pages:
                child_id = child["id"]
                child_title = child.get("properties", {}).get("title", {}).get("title", [{}])[0].get("plain_text", "Untitled")
                click.echo(f"  📄 {child_title} (ID: {child_id})")
        
        click.echo(f"\n✅ Fetch completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Failed to fetch content: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactive mode for collection selection",
)
@click.option(
    "--collection-name",
    "-c",
    help="Collection name",
    default="notion_documents",
    type=str,
)
@click.option(
    "--content-dir",
    "-d",
    help="Directory containing fetched Notion content",
    default="./notion_content",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--chunk-size",
    "-s",
    help="Chunk size in tokens",
    default=512,
    type=int,
)
@click.option(
    "--overlap",
    "-o",
    help="Overlap between chunks in tokens",
    default=50,
    type=int,
)
@click.pass_context
def load(
    ctx: click.Context,
    interactive: bool,
    collection_name: str,
    content_dir: str,
    chunk_size: int,
    overlap: int,
) -> None:
    """Load fetched Notion content into vector database."""
    config = ctx.obj["config"]
    
    try:
        # Initialize vector store
        vector_store = ChromaDBManager(config)
        
        if interactive:
            # Show available collections
            collections = vector_store.list_collections()
            if collections:
                click.echo("📚 Available collections:")
                for i, coll in enumerate(collections, 1):
                    click.echo(f"{i}. {coll['name']} ({coll['count']} documents)")
                
                choice = click.prompt("Select collection or enter new name", type=str)
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(collections):
                        collection_name = collections[choice_num - 1]["name"]
                    else:
                        collection_name = choice
                except ValueError:
                    collection_name = choice
            else:
                collection_name = click.prompt("Enter collection name", default="notion_documents")
        
        # Create or get collection
        collection = vector_store.get_or_create_collection(
            name=collection_name,
            metadata={
                "source": "notion_cli",
                "created_by": "notion-rag-cli",
                "version": "0.1.0"
            }
        )
        
        click.echo(f"📚 Using collection: {collection_name}")
        
        # Check content directory
        content_path = Path(content_dir)
        if not content_path.exists():
            click.echo(f"❌ Content directory not found: {content_path}")
            return
        
        # Find JSON files
        json_files = list(content_path.glob("*.json"))
        if not json_files:
            click.echo(f"❌ No JSON files found in: {content_path}")
            return
        
        click.echo(f"📄 Found {len(json_files)} content files")
        
        total_chunks = 0
        total_pages = 0
        
        for json_file in json_files:
            click.echo(f"\n📄 Processing: {json_file.name}")
            
            # Load JSON content
            import json
            with open(json_file, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            
            page_id = content_data["page_id"]
            title = content_data["title"]
            blocks = content_data["blocks"]
            
            # Extract text from blocks using the improved function
            full_text = extract_text_from_notion_blocks(blocks)
            
            if not full_text.strip():
                click.echo(f"  ⚠️  No text content found")
                continue
        
        # Chunk the text
            chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
        
            # Create document chunks
            doc_chunks = []
        for i, chunk in enumerate(chunks):
            doc_chunk = DocumentChunk(
                id=f"{page_id}_chunk_{i}",
                content=chunk,
                metadata={
                    "source_id": page_id,
                    "source_type": "notion_page",
                    "title": title,
                    "url": f"https://notion.so/{page_id}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_source": json_file.name
                }
            )
            doc_chunks.append(doc_chunk)
            
            # Add to vector store
            success = vector_store.add_documents(collection_name, doc_chunks)
            if success:
                total_pages += 1
                total_chunks += len(doc_chunks)
                click.echo(f"  ✅ Added {len(doc_chunks)} chunks")
            else:
                click.echo(f"  ❌ Failed to add chunks")
        
        # Show final statistics
        final_count = collection.count()
        click.echo(f"\n📊 Loading completed!")
        click.echo(f"  📄 Pages processed: {total_pages}")
        click.echo(f"  📝 Total chunks: {total_chunks}")
        click.echo(f"  📚 Collection documents: {final_count}")
        
        vector_store.close()
        
    except Exception as e:
        click.echo(f"❌ Failed to load content: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactive mode",
)
@click.option(
    "--collection-name",
    "-c",
    help="Collection name to use",
    default="notion_documents",
    type=str,
)
@click.option(
    "--query",
    "-q",
    help="Query to ask",
    type=str,
)
@click.option(
    "--limit",
    "-l",
    help="Maximum number of documents to retrieve",
    default=5,
    type=int,
)
@click.option(
    "--temperature",
    "-t",
    help="Response temperature (0.0 to 2.0)",
    default=0.7,
    type=float,
)
@click.option(
    "--max-tokens",
    "-m",
    help="Maximum tokens to generate",
    default=1000,
    type=int,
)
@click.pass_context
def query(
    ctx: click.Context,
    interactive: bool,
    collection_name: str,
    query: Optional[str],
    limit: int,
    temperature: float,
    max_tokens: int,
) -> None:
    """Query your Notion content using RAG with Gemini."""
    if not check_environment():
        raise click.Abort()
    
    config = ctx.obj["config"]
    
    try:
        # Check for Gemini API key
        if not config.get_gemini_api_key():
            click.echo("❌ GEMINI_API_KEY environment variable is required")
            return
        
        # Initialize components
        vector_store = ChromaDBManager(config)
        gemini_client = create_gemini_client(config)
        
        if interactive:
            # Show available collections
            collections = vector_store.list_collections()
            if collections:
                click.echo("📚 Available collections:")
                for i, coll in enumerate(collections, 1):
                    click.echo(f"{i}. {coll['name']} ({coll['count']} documents)")
                
                choice = click.prompt("Select collection", type=str)
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(collections):
                        collection_name = collections[choice_num - 1]["name"]
                    else:
                        collection_name = choice
                except ValueError:
                    collection_name = choice
            else:
                click.echo("❌ No collections found. Please load content first.")
                return
        
        # Check if collection exists
        collection_info = vector_store.get_collection_info(collection_name)
        if not collection_info:
            click.echo(f"❌ Collection '{collection_name}' not found")
            return
        
        click.echo(f"📚 Using collection: {collection_name} ({collection_info['count']} documents)")
        
        if interactive and not query:
            click.echo("\n💬 Interactive Query Mode")
            click.echo("Type 'exit' to quit, 'help' for commands")
            click.echo("-" * 50)
            
            while True:
                try:
                    user_query = click.prompt("🤔 Your question", type=str)
                    
                    if user_query.lower() in ['exit', 'quit', 'q']:
                        break
                    
                    if user_query.lower() == 'help':
                        click.echo("""
📚 Available Commands:
- help: Show this help message
- stats: Show collection statistics
- exit/quit/q: Exit the program

🎯 Special Query Commands:
- summarize: [question] - Use summarization template
- analyze: [question] - Use content analysis template
- extract: [question] - Use key points extraction template
- bullet: [question] - Use bullet-point summary template
                        """)
                        continue
                    
                    if user_query.lower() == 'stats':
                        click.echo(f"📊 Collection Statistics:")
                        click.echo(f"  Name: {collection_name}")
                        click.echo(f"  Documents: {collection_info['count']}")
                        click.echo(f"  Metadata: {collection_info['metadata']}")
                        continue
                    
                    # Process the query
                    click.echo("🔍 Searching...")
                    
                    # Search for relevant documents
                    results = vector_store.search_documents(
                        collection_name=collection_name,
                        query=user_query,
                        n_results=limit
                    )
                    
                    if not results or not results.get('documents'):
                        click.echo("❌ No relevant documents found")
                        continue
                    
                    # Prepare documents for RAG
                    documents = []
                    for i in range(len(results['documents'][0])):
                        content = results['documents'][0][i]
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        distance = results['distances'][0][i] if results['distances'] else 0.0
                        
                        doc = {
                            "content": content,
                            "metadata": {
                                "title": metadata.get("title", f"Document {i+1}"),
                                "source_id": metadata.get("source_id", "Unknown"),
                                "score": 1.0 - distance
                            }
                        }
                        documents.append(doc)
                    
                    click.echo(f"📚 Found {len(documents)} relevant documents")
                    
                    # Determine prompt template
                    prompt_template = "rag_qa"
                    if user_query.lower().startswith('summarize:'):
                        prompt_template = "rag_summary"
                        user_query = user_query[10:].strip()
                    elif user_query.lower().startswith('analyze:'):
                        prompt_template = "rag_analysis"
                        user_query = user_query[8:].strip()
                    elif user_query.lower().startswith('extract:'):
                        prompt_template = "rag_extraction"
                        user_query = user_query[8:].strip()
                    elif user_query.lower().startswith('bullet:'):
                        prompt_template = "rag_summary"
                        user_query = user_query[7:].strip()
                    
                    # Generate RAG response
                    response = gemini_client.rag_completion(
                        query=user_query,
                        context_documents=documents,
                        prompt_template=prompt_template,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    click.echo(f"\n🤖 Answer:\n{response.content}")
                    click.echo("-" * 50)
                    
                except (KeyboardInterrupt, EOFError):
                    click.echo("\n👋 Goodbye!")
                    sys.exit(0)
                except Exception as e:
                    click.echo(f"❌ Error: {str(e)}")
                    continue
        else:
            # Single query mode
            if not query:
                query = click.prompt("Enter your question")
            
            click.echo(f"🔍 Searching for: '{query}'")
            
            # Search for relevant documents
            results = vector_store.search_documents(
                collection_name=collection_name,
                query=query,
                n_results=limit
            )
            
            if not results or not results.get('documents'):
                click.echo("❌ No relevant documents found")
                return
            
            # Prepare documents for RAG
            documents = []
            for i in range(len(results['documents'][0])):
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0.0
                
                doc = {
                    "content": content,
                    "metadata": {
                        "title": metadata.get("title", f"Document {i+1}"),
                        "source_id": metadata.get("source_id", "Unknown"),
                        "score": 1.0 - distance
                    }
                }
                documents.append(doc)
            
            click.echo(f"📚 Found {len(documents)} relevant documents")
            
            # Generate RAG response
            response = gemini_client.rag_completion(
                query=query,
                context_documents=documents,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            click.echo(f"\n🤖 Answer:\n{response.content}")
        
        # Cleanup
        vector_store.close()
        gemini_client.close()
        
    except Exception as e:
        click.echo(f"❌ Query failed: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--collection-name",
    "-c",
    help="Collection name to search",
    required=True,
    type=str,
)
@click.option(
    "--query",
    "-q",
    help="Query to ask",
    required=True,
    type=str,
)
@click.option(
    "--limit",
    "-l",
    help="Maximum number of documents to retrieve",
    default=5,
    type=int,
)
@click.option(
    "--temperature",
    "-t",
    help="Response temperature (0.0 to 2.0)",
    default=0.7,
    type=float,
)
@click.option(
    "--max-tokens",
    "-m",
    help="Maximum tokens to generate",
    default=1000,
    type=int,
)
@click.pass_context
def rag_query(
    ctx: click.Context,
    collection_name: str,
    query: str,
    limit: int,
    temperature: float,
    max_tokens: int,
) -> None:
    """Query documents using RAG with Gemini."""
    config = ctx.obj["config"]
    
    try:
        click.echo(f"🔍 Searching collection: {collection_name}")
        click.echo(f"❓ Query: {query}")
        
        # Initialize vector store
        vector_store = ChromaDBManager(config)
        
        # Search for relevant documents
        results = vector_store.search_similar(
            collection_name=collection_name,
            query_text=query,
            n_results=limit
        )
        
        if not results:
            click.echo("❌ No relevant documents found")
            return
        
        click.echo(f"📚 Found {len(results)} relevant documents")
        
        # Initialize Gemini client
        click.echo("🤖 Initializing Gemini client...")
        client = create_gemini_client(config)
        
        # Prepare documents for RAG
        documents = []
        for i, result in enumerate(results):
            doc = {
                "content": result.content,
                "metadata": {
                    "title": f"Document {i+1}",
                    "source_id": result.metadata.get("source_id", "unknown"),
                    "similarity_score": result.similarity_score
                }
            }
            documents.append(doc)
            click.echo(f"  📄 Document {i+1}: {result.content[:100]}...")
        
        # Generate RAG response
        click.echo("🧠 Generating response with Gemini...")
        response = client.rag_completion(
            query=query,
            context_documents=documents,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        click.echo("\n" + "="*50)
        click.echo("🤖 GEMINI RESPONSE:")
        click.echo("="*50)
        click.echo(response.content)
        click.echo("="*50)
        click.echo(f"📊 Usage: {response.usage}")
        click.echo(f"🔧 Model: {response.model}")
        
        # Cleanup
        vector_store.close()
        client.close()
        
    except Exception as e:
        click.echo(f"❌ RAG query failed: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--collection-name",
    "-c",
    help="Collection name to use",
    default="notion_documents",
    type=str,
)
@click.pass_context
def rag_chat(ctx: click.Context, collection_name: str) -> None:
    """Start interactive RAG chat with Gemini."""
    config = ctx.obj["config"]
    
    try:
        click.echo("🔧 Initializing RAG chat system...")
        
        # Initialize components
        vector_store = ChromaDBManager(config)
        client = create_gemini_client(config)
        
        # Check if collection exists
        collections = vector_store.list_collections()
        collection_names = [c['name'] for c in collections]
        
        if collection_name not in collection_names:
            click.echo(f"❌ Collection '{collection_name}' not found")
            click.echo(f"Available collections: {collection_names}")
            return
        
        click.echo(f"✅ Connected to collection: {collection_name}")
        click.echo(f"📊 Documents in collection: {vector_store.get_collection_info(collection_name)['count']}")
        click.echo("💬 Start chatting! Type 'exit' to quit.")
        click.echo("-" * 50)
        
        while True:
            try:
                query = input("🤖 You: ")
                if query.lower() in ['exit', 'quit', 'bye']:
                    break
                
                if not query.strip():
                    continue
                
                click.echo("🔍 Searching for relevant documents...")
                
                # Search for relevant documents
                results = vector_store.search_similar(
                    collection_name=collection_name,
                    query_text=query,
                    n_results=5
                )
                
                if not results:
                    click.echo("❌ No relevant documents found")
                    continue
                
                # Prepare documents for RAG
                documents = []
                for i, result in enumerate(results):
                    doc = {
                        "content": result.content,
                        "metadata": {
                            "title": f"Document {i+1}",
                            "source_id": result.metadata.get("source_id", "unknown"),
                            "similarity_score": result.similarity_score
                        }
                    }
                    documents.append(doc)
                
                click.echo(f"📚 Found {len(documents)} relevant documents")
                click.echo("🧠 Generating response...")
                
                # Generate RAG response
                response = client.rag_completion(
                    query=query,
                    context_documents=documents,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                click.echo(f"🤖 Gemini: {response.content}")
                click.echo("-" * 50)
                
            except (KeyboardInterrupt, EOFError):
                click.echo("\n👋 Goodbye!")
                sys.exit(0)
            except Exception as e:
                click.echo(f"❌ Error: {str(e)}")
                continue
        
        # Cleanup
        vector_store.close()
        client.close()
        
    except Exception as e:
        click.echo(f"❌ Failed to start RAG chat: {str(e)}")
        raise click.Abort()


@cli.command()
@click.pass_context
def collections(ctx: click.Context) -> None:
    """List all ChromaDB collections."""
    config = ctx.obj["config"]
    
    try:
        vector_store = ChromaDBManager(config)
        collections = vector_store.list_collections()
        
        if not collections:
            click.echo("📭 No collections found")
            return
        
        click.echo("📚 Available collections:")
        for collection in collections:
            click.echo(f"  • {collection['name']} ({collection['count']} documents)")
            if collection['metadata']:
                click.echo(f"    Metadata: {collection['metadata']}")
        
        vector_store.close()
        
    except Exception as e:
        click.echo(f"❌ Failed to list collections: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--collection-name",
    "-c",
    help="Collection name",
    required=True,
    type=str,
)
@click.pass_context
def collection_info(ctx: click.Context, collection_name: str) -> None:
    """Get detailed information about a collection."""
    config = ctx.obj["config"]
    
    try:
        vector_store = ChromaDBManager(config)
        info = vector_store.get_collection_info(collection_name)
        
        if not info:
            click.echo(f"❌ Collection '{collection_name}' not found")
            return
        
        click.echo(f"📊 Collection: {info['name']}")
        click.echo(f"📄 Document count: {info['count']}")
        click.echo(f"🏷️  Metadata: {info['metadata']}")
        
        # Get additional stats
        stats = vector_store.get_collection_stats(collection_name)
        if stats:
            click.echo(f"📈 Total characters: {stats['total_chars']}")
            click.echo(f"📏 Average chars per doc: {stats['avg_chars_per_doc']}")
        
        vector_store.close()
        
    except Exception as e:
        click.echo(f"❌ Failed to get collection info: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--collection-name",
    "-c",
    help="Collection name",
    required=True,
    type=str,
)
@click.pass_context
def clear_collection(ctx: click.Context, collection_name: str) -> None:
    """Clear all documents from a collection."""
    config = ctx.obj["config"]
    
    if not click.confirm(f"Are you sure you want to clear collection '{collection_name}'?"):
        click.echo("Operation cancelled")
        return
    
    try:
        vector_store = ChromaDBManager(config)
        success = vector_store.clear_collection(collection_name)
        
        if success:
            click.echo(f"✅ Collection '{collection_name}' cleared successfully")
        else:
            click.echo(f"❌ Failed to clear collection '{collection_name}'")
        
        vector_store.close()
        
    except Exception as e:
        click.echo(f"❌ Failed to clear collection: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--collection-name",
    "-c",
    help="Collection name",
    required=True,
    type=str,
)
@click.pass_context
def delete_collection(ctx: click.Context, collection_name: str) -> None:
    """Delete a collection completely."""
    config = ctx.obj["config"]
    
    if not click.confirm(f"Are you sure you want to delete collection '{collection_name}'? This action cannot be undone."):
        click.echo("Operation cancelled")
        return
    
    try:
        vector_store = ChromaDBManager(config)
        success = vector_store.delete_collection(collection_name)
        
        if success:
            click.echo(f"✅ Collection '{collection_name}' deleted successfully")
        else:
            click.echo(f"❌ Failed to delete collection '{collection_name}'")
        
        vector_store.close()
        
    except Exception as e:
        click.echo(f"❌ Failed to delete collection: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--api-key",
    "-k",
    help="Gemini API key (optional, will use environment variable if not provided)",
    type=str,
)
@click.pass_context
def gemini_test(ctx: click.Context, api_key: str) -> None:
    """Test Gemini API connection and basic functionality."""
    config = ctx.obj["config"]
    
    try:
        click.echo("🔧 Initializing Gemini client...")
        client = create_gemini_client(config, api_key)
        
        # Test connection
        click.echo("🔗 Testing connection...")
        if client.test_connection():
            click.echo("✅ Connection successful!")
        else:
            click.echo("❌ Connection failed!")
            return
        
        # Test basic chat completion
        click.echo("💬 Testing chat completion...")
        messages = [GeminiMessage("user", "Hello! Please respond with 'Hello from Gemini!'")]
        
        response = client.chat_completion(messages, temperature=0.1)
        click.echo(f"🤖 Response: {response.content}")
        click.echo(f"📊 Usage: {response.usage}")
        
        # Test available models
        click.echo("📋 Available models:")
        models = client.get_available_models()
        for model in models[:3]:  # Show first 3 models
            click.echo(f"  • {model.get('name', 'Unknown')}")
        
        client.close()
        click.echo("✅ Gemini test completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Gemini test failed: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactive mode",
)
@click.option(
    "--page-id",
    "-p",
    help="Specific Notion page ID to process",
    type=str,
)
@click.option(
    "--collection-name",
    "-c",
    help="Collection name to use",
    default="notion_documents",
    type=str,
)
@click.option(
    "--output-dir",
    "-o",
    help="Output directory for fetched content",
    default="./notion_content",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.pass_context
def workflow(
    ctx: click.Context,
    interactive: bool,
    page_id: Optional[str],
    collection_name: str,
    output_dir: str,
) -> None:
    """Complete workflow: fetch → load → query (interactive)."""
    if not check_environment():
        raise click.Abort()
    
    config = ctx.obj["config"]
    
    try:
        click.echo("🚀 Starting Notion RAG Workflow")
        click.echo("=" * 50)
        
        # Step 1: Fetch content
        click.echo("\n📥 Step 1: Fetching Notion content...")
        notion_client = get_notion_client()
        if not notion_client:
            raise click.Abort()
        
        if interactive:
            # Interactive page selection
            click.echo("🔍 Interactive Notion Page Selection")
            click.echo("=" * 30)
            
            # Get home page info
            home_page_id = os.getenv("NOTION_HOME_PAGE_ID")
            click.echo(f"🏠 Home page ID: {home_page_id}")
            
            # Show available options
            click.echo("\n📋 Available options:")
            click.echo("1. Fetch home page only")
            click.echo("2. Fetch home page with all children")
            click.echo("3. Enter specific page ID")
            click.echo("4. Search for pages")
            
            choice = click.prompt("\nSelect option", type=click.Choice(["1", "2", "3", "4"]))
            
            if choice == "1":
                page_id = home_page_id
                recursive = False
            elif choice == "2":
                page_id = home_page_id
                recursive = True
            elif choice == "3":
                page_id = click.prompt("Enter page ID")
                recursive = click.confirm("Fetch child pages recursively?")
            elif choice == "4":
                query = click.prompt("Enter search query")
                pages = notion_client.search_pages(query)
                if pages:
                    click.echo(f"\n📄 Found {len(pages)} pages:")
                    for i, page in enumerate(pages[:10], 1):
                        title = page.get("properties", {}).get("title", {}).get("title", [{}])[0].get("plain_text", "Untitled")
                        click.echo(f"{i}. {title} (ID: {page['id']})")
                    
                    if len(pages) > 10:
                        click.echo(f"... and {len(pages) - 10} more pages")
                    
                    page_choice = click.prompt("Select page number", type=int, default=1)
                    if 1 <= page_choice <= len(pages):
                        page_id = pages[page_choice - 1]["id"]
                        recursive = click.confirm("Fetch child pages recursively?")
                    else:
                        click.echo("❌ Invalid selection")
                        return
                else:
                    click.echo("❌ No pages found")
                    return
        else:
            if not page_id:
                page_id = os.getenv("NOTION_HOME_PAGE_ID")
            recursive = click.confirm("Fetch child pages recursively?")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"🚀 Fetching page: {page_id}")
        if recursive:
            click.echo("📂 Including all child pages")
        
        # Fetch page content
        page_content = notion_client.get_page_content(page_id)
        
        # Extract page title
        title = "Untitled"
        if "properties" in page_content:
            title_prop = page_content["properties"].get("title", {})
            if "title" in title_prop:
                title = title_prop["title"][0]["plain_text"]
        
        click.echo(f"📄 Page title: {title}")
        
        # Get page blocks
        blocks = notion_client.get_block_children(page_id)
        click.echo(f"📝 Found {len(blocks)} blocks")
        
        # Save content
        content_file = output_path / f"{page_id}.json"
        import json
        with open(content_file, 'w', encoding='utf-8') as f:
            json.dump({
                "page_id": page_id,
                "title": title,
                "blocks": blocks,
                "recursive": recursive
            }, f, indent=2, ensure_ascii=False)
        
        click.echo(f"✅ Content saved to: {content_file}")
        
        if recursive:
            # Get child pages
            child_pages = notion_client.get_all_child_pages(page_id)
            click.echo(f"📂 Found {len(child_pages)} child pages")
            
            for child in child_pages:
                child_id = child["id"]
                child_title = child.get("properties", {}).get("title", {}).get("title", [{}])[0].get("plain_text", "Untitled")
                click.echo(f"  📄 {child_title} (ID: {child_id})")
        
        # Step 2: Load into vector database
        click.echo("\n📚 Step 2: Loading into vector database...")
        
        # Initialize vector store
        vector_store = ChromaDBManager(config)
        
        # Create or get collection
        collection = vector_store.get_or_create_collection(
            name=collection_name,
            metadata={
                "source": "notion_workflow",
                "created_by": "notion-rag-cli",
                "version": "0.1.0"
            }
        )
        
        click.echo(f"📚 Using collection: {collection_name}")
        
        # Find JSON files
        json_files = list(output_path.glob("*.json"))
        if not json_files:
            click.echo(f"❌ No JSON files found in: {output_path}")
            return
        
        click.echo(f"📄 Found {len(json_files)} content files")
        
        total_chunks = 0
        total_pages = 0
        
        for json_file in json_files:
            click.echo(f"\n📄 Processing: {json_file.name}")
            
            # Load JSON content
            with open(json_file, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            
            page_id = content_data["page_id"]
            title = content_data["title"]
            blocks = content_data["blocks"]
            
            # Extract text from blocks using the improved function
            full_text = extract_text_from_notion_blocks(blocks)
            
            if not full_text.strip():
                click.echo(f"  ⚠️  No text content found")
                continue
            
            # Chunk the text
            chunks = chunk_text(full_text, chunk_size=512, overlap=50)
            
            # Create document chunks
            doc_chunks = []
            for i, chunk in enumerate(chunks):
                doc_chunk = DocumentChunk(
                    id=f"{page_id}_chunk_{i}",
                    content=chunk,
                    metadata={
                        "source_id": page_id,
                        "source_type": "notion_page",
                        "title": title,
                        "url": f"https://notion.so/{page_id}",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_source": json_file.name
                    }
                )
                doc_chunks.append(doc_chunk)
            
            # Add to vector store
            success = vector_store.add_documents(collection_name, doc_chunks)
            if success:
                total_pages += 1
                total_chunks += len(doc_chunks)
                click.echo(f"  ✅ Added {len(doc_chunks)} chunks")
            else:
                click.echo(f"  ❌ Failed to add chunks")
        
        # Show final statistics
        final_count = collection.count()
        click.echo(f"\n📊 Loading completed!")
        click.echo(f"  📄 Pages processed: {total_pages}")
        click.echo(f"  📝 Total chunks: {total_chunks}")
        click.echo(f"  📚 Collection documents: {final_count}")
        
        # Step 3: Interactive query
        click.echo("\n💬 Step 3: Interactive Query Mode")
        click.echo("=" * 30)
        
        # Check for Gemini API key
        if not config.get_gemini_api_key():
            click.echo("❌ GEMINI_API_KEY environment variable is required for querying")
            click.echo("💡 Set it to enable RAG queries with Gemini")
            vector_store.close()
            return
        
        # Initialize Gemini client
        gemini_client = create_gemini_client(config)
        
        click.echo("💬 Start asking questions about your Notion content!")
        click.echo("Type 'exit' to quit, 'help' for commands")
        click.echo("-" * 50)
        
        while True:
            try:
                user_query = click.prompt("🤔 Your question", type=str)
                
                if user_query.lower() in ['exit', 'quit', 'q']:
                    break
                
                if user_query.lower() == 'help':
                    click.echo("""
📚 Available Commands:
- help: Show this help message
- stats: Show collection statistics
- exit/quit/q: Exit the program

🎯 Special Query Commands:
- summarize: [question] - Use summarization template
- analyze: [question] - Use content analysis template
- extract: [question] - Use key points extraction template
- bullet: [question] - Use bullet-point summary template
                    """)
                    continue
                
                if user_query.lower() == 'stats':
                    collection_info = vector_store.get_collection_info(collection_name)
                    if collection_info:
                        click.echo(f"📊 Collection Statistics:")
                        click.echo(f"  Name: {collection_name}")
                        click.echo(f"  Documents: {collection_info['count']}")
                        click.echo(f"  Metadata: {collection_info['metadata']}")
                    continue
                
                # Process the query
                click.echo("🔍 Searching...")
                
                # Search for relevant documents
                results = vector_store.search_documents(
                    collection_name=collection_name,
                    query=user_query,
                    n_results=5
                )
                
                if not results or not results.get('documents'):
                    click.echo("❌ No relevant documents found")
                    continue
                
                # Prepare documents for RAG
                documents = []
                for i in range(len(results['documents'][0])):
                    content = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0.0
                    
                    doc = {
                        "content": content,
                        "metadata": {
                            "title": metadata.get("title", f"Document {i+1}"),
                            "source_id": metadata.get("source_id", "Unknown"),
                            "score": 1.0 - distance
                        }
                    }
                    documents.append(doc)
                
                click.echo(f"📚 Found {len(documents)} relevant documents")
                
                # Determine prompt template
                prompt_template = "rag_qa"
                if user_query.lower().startswith('summarize:'):
                    prompt_template = "rag_summary"
                    user_query = user_query[10:].strip()
                elif user_query.lower().startswith('analyze:'):
                    prompt_template = "rag_analysis"
                    user_query = user_query[8:].strip()
                elif user_query.lower().startswith('extract:'):
                    prompt_template = "rag_extraction"
                    user_query = user_query[8:].strip()
                elif user_query.lower().startswith('bullet:'):
                    prompt_template = "rag_summary"
                    user_query = user_query[7:].strip()
                
                # Generate RAG response
                response = gemini_client.rag_completion(
                    query=user_query,
                    context_documents=documents,
                    prompt_template=prompt_template,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                click.echo(f"\n🤖 Answer:\n{response.content}")
                click.echo("-" * 50)
                
            except (KeyboardInterrupt, EOFError):
                click.echo("\n👋 Goodbye!")
                sys.exit(0)
            except Exception as e:
                click.echo(f"❌ Error: {str(e)}")
                continue
        
        # Cleanup
        vector_store.close()
        gemini_client.close()
        click.echo("\n✅ Workflow completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Workflow failed: {str(e)}")
        raise click.Abort()


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status and configuration."""
    config = ctx.obj["config"]
    
    click.echo("🔍 Notion RAG System Status")
    click.echo("=" * 40)
    
    # Environment variables
    click.echo("\n📋 Environment Variables:")
    notion_key = os.getenv("NOTION_API_KEY")
    notion_page = os.getenv("NOTION_HOME_PAGE_ID")
    gemini_key = os.getenv("GEMINI_API_KEY")
    google_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    
    click.echo(f"  NOTION_API_KEY: {'✅ Set' if notion_key else '❌ Missing'}")
    click.echo(f"  NOTION_HOME_PAGE_ID: {'✅ Set' if notion_page else '❌ Missing'}")
    click.echo(f"  GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Missing'}")
    click.echo(f"  GOOGLE_CLOUD_PROJECT: {'✅ Set' if google_project else '❌ Missing'}")
    
    # Vector database status
    click.echo("\n📚 Vector Database:")
    try:
        vector_store = ChromaDBManager(config)
        collections = vector_store.list_collections()
        
        if collections:
            click.echo(f"  Status: ✅ Connected")
            click.echo(f"  Collections: {len(collections)}")
            for coll in collections:
                click.echo(f"    • {coll['name']} ({coll['count']} documents)")
        else:
            click.echo(f"  Status: ✅ Connected (no collections)")
        
        vector_store.close()
    except Exception as e:
        click.echo(f"  Status: ❌ Error: {str(e)}")
    
    # Gemini status
    click.echo("\n🤖 Gemini API:")
    if gemini_key:
        try:
            gemini_client = create_gemini_client(config)
            if gemini_client.test_connection():
                click.echo("  Status: ✅ Connected")
            else:
                click.echo("  Status: ❌ Connection failed")
            gemini_client.close()
        except Exception as e:
            click.echo(f"  Status: ❌ Error: {str(e)}")
    else:
        click.echo("  Status: ⚠️  API key not set")
    
    # Configuration
    click.echo("\n⚙️  Configuration:")
    click.echo(f"  ChromaDB Path: {config.get_chroma_db_path()}")
    click.echo(f"  Default Collection: {config.chroma.collection_name}")
    click.echo(f"  Log Level: {config.log_level}")
    click.echo(f"  Max Retries: {config.max_retries}")


@cli.command()
@click.pass_context
def setup(ctx: click.Context) -> None:
    """Interactive setup wizard for the Notion RAG system."""
    click.echo("🔧 Notion RAG Setup Wizard")
    click.echo("=" * 40)
    
    click.echo("\nThis wizard will help you set up the Notion RAG system.")
    click.echo("You'll need:")
    click.echo("• A Notion API key")
    click.echo("• A Notion page ID")
    click.echo("• A Gemini API key")
    click.echo("• A Google Cloud project ID")
    
    if not click.confirm("\nDo you want to continue?"):
        click.echo("Setup cancelled.")
        return
    
    # Check current environment
    click.echo("\n📋 Current Environment Status:")
    notion_key = os.getenv("NOTION_API_KEY")
    notion_page = os.getenv("NOTION_HOME_PAGE_ID")
    gemini_key = os.getenv("GEMINI_API_KEY")
    google_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    
    click.echo(f"  NOTION_API_KEY: {'✅ Set' if notion_key else '❌ Missing'}")
    click.echo(f"  NOTION_HOME_PAGE_ID: {'✅ Set' if notion_page else '❌ Missing'}")
    click.echo(f"  GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Missing'}")
    click.echo(f"  GOOGLE_CLOUD_PROJECT: {'✅ Set' if google_project else '❌ Missing'}")
    
    # Guide for missing variables
    missing_vars = []
    if not notion_key:
        missing_vars.append("NOTION_API_KEY")
    if not notion_page:
        missing_vars.append("NOTION_HOME_PAGE_ID")
    if not gemini_key:
        missing_vars.append("GEMINI_API_KEY")
    if not google_project:
        missing_vars.append("GOOGLE_CLOUD_PROJECT")
    
    if missing_vars:
        click.echo(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        click.echo("\n💡 To set these variables:")
        click.echo("1. Create a .env file in the project root")
        click.echo("2. Add the following lines:")
        click.echo("   NOTION_API_KEY=your_notion_api_key")
        click.echo("   NOTION_HOME_PAGE_ID=your_notion_page_id")
        click.echo("   GEMINI_API_KEY=your_gemini_api_key")
        click.echo("   GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id")
        click.echo("\n3. Restart the application")
    else:
        click.echo("\n✅ All required environment variables are set!")
        
        # Test connections
        click.echo("\n🔗 Testing connections...")
        
        # Test Notion
        try:
            notion_client = get_notion_client()
            if notion_client:
                click.echo("  ✅ Notion API: Connected")
            else:
                click.echo("  ❌ Notion API: Failed")
        except Exception as e:
            click.echo(f"  ❌ Notion API: Error - {str(e)}")
        
        # Test Gemini
        if gemini_key:
            try:
                config = ctx.obj["config"]
                gemini_client = create_gemini_client(config)
                if gemini_client.test_connection():
                    click.echo("  ✅ Gemini API: Connected")
                else:
                    click.echo("  ❌ Gemini API: Connection failed")
                gemini_client.close()
            except Exception as e:
                click.echo(f"  ❌ Gemini API: Error - {str(e)}")
        
        # Test Vector Database
        try:
            config = ctx.obj["config"]
            vector_store = ChromaDBManager(config)
            collections = vector_store.list_collections()
            click.echo(f"  ✅ Vector Database: Connected ({len(collections)} collections)")
            vector_store.close()
        except Exception as e:
            click.echo(f"  ❌ Vector Database: Error - {str(e)}")
        
        click.echo("\n🎉 Setup completed! You can now use the system.")
        click.echo("\nNext steps:")
        click.echo("1. Run 'notion-rag fetch -i' to fetch content")
        click.echo("2. Run 'notion-rag load -i' to load into vector database")
        click.echo("3. Run 'notion-rag query -i' to start querying")
        click.echo("4. Or run 'notion-rag workflow -i' for the complete workflow")


def main() -> None:
    """Main entry point for the CLI."""
    cli() 