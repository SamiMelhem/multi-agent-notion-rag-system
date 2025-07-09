"""
Command-line interface for the Notion RAG system.
"""

import click
from typing import Optional

from .config import Config


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
    # TODO: Add initialization logic


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


def main() -> None:
    """Main entry point for the CLI."""
    cli() 