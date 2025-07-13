"""
Command-line interface for the Notion RAG system.
"""

import click
from .config import Config
from .vector_store import ChromaDBManager, DocumentChunk
from .embeddings import create_embedding_generator
from .chunking import chunk_text, count_tokens


@click.group()
@click.version_option(version="0.1.0")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    Notion RAG CLI - A multi-agent RAG system for Notion API.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config()


@cli.command()
@click.option(
    "--database-id",
    "-d",
    help="Notion database ID to process",
    required=True,
    type=str,
)
@click.option(
    "--collection-name",
    "-c",
    help="ChromaDB collection name",
    default="notion_documents",
    type=str,
)
@click.pass_context
def init(
    ctx: click.Context,
    database_id: str,
    collection_name: str,
) -> None:
    """Initialize the RAG system with configuration."""
    config = ctx.obj["config"]
    click.echo(f"Initializing RAG system with database: {database_id}")
    click.echo(f"Collection name: {collection_name}")
    
    try:
        # Initialize ChromaDB
        vector_store = ChromaDBManager(config)
        
        # Create collection
        collection = vector_store.get_or_create_collection(
            name=collection_name,
            metadata={
                "database_id": database_id,
                "created_by": "notion-rag-cli",
                "version": "0.1.0"
            }
        )
        
        click.echo(f"✅ Collection '{collection_name}' initialized successfully")
        click.echo(f"📊 Collection info: {collection.count()} documents")
        
        vector_store.close()
        
    except Exception as e:
        click.echo(f"❌ Failed to initialize: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--text",
    "-t",
    help="Text to chunk",
    required=True,
    type=str,
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
def chunk(ctx: click.Context, text: str, chunk_size: int, overlap: int) -> None:
    """Chunk text into overlapping segments."""
    config = ctx.obj["config"]
    
    try:
        click.echo(f"📝 Chunking text (size: {chunk_size}, overlap: {overlap})...")
        
        # Count tokens first
        token_count = count_tokens(text)
        click.echo(f"📊 Total tokens: {token_count}")
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
        click.echo(f"✅ Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            chunk_tokens = count_tokens(chunk)
            click.echo(f"  Chunk {i+1}: {chunk_tokens} tokens")
            click.echo(f"    Preview: {chunk[:100]}...")
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Failed to chunk text: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--text",
    "-t",
    help="Text to generate embedding for",
    required=True,
    type=str,
)
@click.option(
    "--model",
    "-m",
    help="Model name to use",
    default="BAAI/bge-small-en-v1.5",
    type=str,
)
@click.pass_context
def embed(ctx: click.Context, text: str, model: str) -> None:
    """Generate embeddings for text."""
    config = ctx.obj["config"]
    
    try:
        click.echo(f"🔧 Initializing embedding generator with model: {model}")
        
        # Create embedding generator
        generator = create_embedding_generator(config, model)
        
        # Get model info
        info = generator.get_model_info()
        click.echo(f"📊 Model info:")
        click.echo(f"  Name: {info['model_name']}")
        click.echo(f"  Dimension: {info['embedding_dimension']}")
        click.echo(f"  Type: {info['model_type']}")
        
        # Generate embedding
        click.echo(f"🔄 Generating embedding for text...")
        embedding = generator.generate_single_embedding(text)
        
        click.echo(f"✅ Generated embedding with {len(embedding)} dimensions")
        click.echo(f"📈 First 5 values: {embedding[:5]}")
        
    except ImportError as e:
        click.echo(f"❌ Missing dependencies: {str(e)}")
        click.echo("💡 Install with: pip install sentence-transformers")
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Failed to generate embedding: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--collection-name",
    "-c",
    help="Collection name",
    required=True,
    type=str,
)
@click.option(
    "--text",
    "-t",
    help="Text to add",
    required=True,
    type=str,
)
@click.option(
    "--source-id",
    "-s",
    help="Source ID for the text",
    default="test",
    type=str,
)
@click.pass_context
def add_text(ctx: click.Context, collection_name: str, text: str, source_id: str) -> None:
    """Add text to collection with automatic chunking and embedding."""
    config = ctx.obj["config"]
    
    try:
        click.echo(f"📚 Adding text to collection: {collection_name}")
        
        # Initialize vector store
        vector_store = ChromaDBManager(config)
        
        # Prepare metadata
        metadata = {
            "source_id": source_id,
            "source_type": "text",
            "created_by": "cli"
        }
        
        # Add text as chunks
        success = vector_store.add_text_as_chunks(
            collection_name=collection_name,
            text=text,
            metadata=metadata
        )
        
        if success:
            click.echo(f"✅ Text added successfully to collection {collection_name}")
            
            # Get collection info
            info = vector_store.get_collection_info(collection_name)
            if info:
                click.echo(f"📊 Collection now has {info['count']} documents")
        else:
            click.echo(f"❌ Failed to add text to collection")
        
        vector_store.close()
        
    except Exception as e:
        click.echo(f"❌ Failed to add text: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option(
    "--database-id",
    "-d",
    help="Notion database ID to index",
    required=True,
    type=str,
)
@click.option(
    "--batch-size",
    "-b",
    help="Batch size for processing",
    default=100,
    type=int,
)
@click.pass_context
def index(
    ctx: click.Context,
    database_id: str,
    batch_size: int,
) -> None:
    """Index Notion pages for search."""
    config = ctx.obj["config"]
    click.echo(f"Indexing database: {database_id}")
    click.echo(f"Batch size: {batch_size}")
    # TODO: Add indexing logic


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--limit",
    "-l",
    help="Maximum number of results",
    default=10,
    type=int,
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    limit: int,
) -> None:
    """Search documents using natural language query."""
    config = ctx.obj["config"]
    click.echo(f"Searching for: '{query}'")
    click.echo(f"Limit: {limit}")
    # TODO: Add search logic


@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Start interactive chat with your documents."""
    config = ctx.obj["config"]
    click.echo("Starting interactive chat mode...")
    click.echo("Type 'exit' to quit.")
    
    while True:
        try:
            query = input("🤖 Ask about your documents: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            # TODO: Add chat logic
            click.echo(f"You asked: {query}")
            click.echo("This feature is not yet implemented.")
            
        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            break


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


def main() -> None:
    """Main entry point for the CLI."""
    cli() 