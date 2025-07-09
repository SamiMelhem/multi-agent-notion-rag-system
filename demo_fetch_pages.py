#!/usr/bin/env python3
"""
Demo script to fetch and display pages from Notion.
This script demonstrates how to use the NotionClient to fetch pages and databases.
"""

import json
from typing import Dict, Any, List
from notion_rag.notion_client import NotionClient


def print_page_info(page: Dict[str, Any], indent: int = 0) -> None:
    """Print formatted page information."""
    indent_str = "  " * indent
    
    # Extract title from page properties
    title = "Untitled"
    if "properties" in page and "title" in page["properties"]:
        title_prop = page["properties"]["title"]
        if "title" in title_prop and title_prop["title"]:
            title = title_prop["title"][0]["text"]["content"]
    
    print(f"{indent_str}📄 {title}")
    print(f"{indent_str}   ID: {page['id']}")
    print(f"{indent_str}   URL: {page.get('url', 'N/A')}")
    print(f"{indent_str}   Type: {page.get('type', 'N/A')}")
    
    # Print some properties if available
    if "properties" in page:
        print(f"{indent_str}   Properties:")
        for prop_name, prop_value in page["properties"].items():
            if prop_name != "title":  # Skip title as we already printed it
                prop_type = prop_value.get("type", "unknown")
                print(f"{indent_str}     - {prop_name} ({prop_type})")
    
    print()


def print_database_info(database: Dict[str, Any], indent: int = 0) -> None:
    """Print formatted database information."""
    indent_str = "  " * indent
    
    # Extract title from database properties
    title = "Untitled Database"
    if "title" in database and database["title"]:
        title = database["title"][0]["text"]["content"]
    
    print(f"{indent_str}🗃️  {title}")
    print(f"{indent_str}   ID: {database['id']}")
    print(f"{indent_str}   URL: {database.get('url', 'N/A')}")
    print(f"{indent_str}   Type: {database.get('type', 'N/A')}")
    
    # Print database properties
    if "properties" in database:
        print(f"{indent_str}   Properties:")
        for prop_name, prop_value in database["properties"].items():
            prop_type = prop_value.get("type", "unknown")
            print(f"{indent_str}     - {prop_name} ({prop_type})")
    
    print()


def format_rich_text_to_markdown(rich_text_list):
    md = ""
    for text_obj in rich_text_list:
        text = text_obj.get("plain_text", "")
        ann = text_obj.get("annotations", {})
        # Markdown formatting
        if ann.get("code"):
            text = f'`{text}`'
        if ann.get("bold"):
            text = f'**{text}**'
        if ann.get("italic"):
            text = f'*{text}*'
        if ann.get("strikethrough"):
            text = f'~~{text}~~'
        if ann.get("underline"):
            text = f'_{text}_'  # Markdown doesn't support underline natively
        md += text
    return md

def extract_and_print_markdown_from_page(page_id, client):
    # 1. Print the home page title
    metadata = client.get_page_content(page_id)
    title = metadata["properties"]["title"]["title"][0]["text"]["content"]
    print(f"# {title}\n")

    # 2. List sub-page titles
    blocks = client.get_block_children(page_id)
    sub_page_titles = [block["child_page"]["title"] for block in blocks if block.get("type") == "child_page"]
    if sub_page_titles:
        for t in sub_page_titles:
            print(f"- {t}")
    else:
        print("(No sub-pages found)\n")

    # 3. Markdown-formatted bullet list of non-page blocks
    for block in blocks:
        if block.get("type") != "child_page":
            block_type = block["type"]
            rich_text_list = block.get(block_type, {}).get("rich_text", [])
            if rich_text_list:
                md = format_rich_text_to_markdown(rich_text_list)
                print(f"- {md}")
    print()


def demo_fetch_pages():
    """Demonstrate fetching pages from Notion."""
    try:
        print("🚀 Initializing Notion Client...")
        client = NotionClient()
        print("✅ Notion Client initialized successfully!\n")

        # # Print the metadata for the home page
        # print("================ HOME PAGE METADATA ================")
        # home_page_id = client.home_page_id
        # home_page_metadata = client.get_page_content(home_page_id)
        # print(json.dumps(home_page_metadata, indent=2, default=str))
        # print("====================================================\n")

        # # Print the blocks (content) for the home page
        # print("================ HOME PAGE BLOCKS ================" )
        # blocks = client.get_block_children(home_page_id)
        # print(json.dumps(blocks, indent=2, default=str))
        # print("==================================================\n")

        # For the first sub-page, print its blocks (commented out for now)
        # first_sub_page = None
        # for block in blocks:
        #     if block.get("type") == "child_page":
        #         first_sub_page = block
        #         break
        # if first_sub_page:
        #     sub_page_id = first_sub_page["id"]
        #     sub_page_title = first_sub_page["child_page"].get("title", "Untitled")
        #     print(f"==== BLOCKS FOR SUB-PAGE: {sub_page_title} ({sub_page_id}) ====")
        #     sub_page_blocks = client.get_block_children(sub_page_id)
        #     print(json.dumps(sub_page_blocks, indent=2, default=str))
        #     print("============================================\n")

        # Call the new markdown extraction function
        print("\n================ MARKDOWN EXTRACTION ================")
        extract_and_print_markdown_from_page(client.home_page_id, client)
        print("====================================================\n")

        # # 1. Get all child pages from home page
        # print("📋 Fetching all child pages from home page...")
        # print("[INFO] Calling get_all_child_pages() ...")
        # child_pages = client.get_all_child_pages()
        # print(f"[INFO] get_all_child_pages() returned {len(child_pages)} items.")
        # print(f"Found {len(child_pages)} child pages/databases\n")
        
        # # 2. Display child pages and databases
        # for idx, item in enumerate(child_pages):
        #     print(f"[INFO] Processing child {idx+1}/{len(child_pages)}: {item.get('id', 'N/A')}")
        #     if item["type"] == "child_page":
        #         print_page_info(item)
        #     elif item["type"] == "child_database":
        #         print_database_info(item)
        
        # # 3. If there are child pages, get content of the first page
        # if child_pages:
        #     first_item = child_pages[0]
        #     if first_item["type"] == "child_page":
        #         print(f"🔍 Getting detailed content for first page... (ID: {first_item['id']})")
        #         try:
        #             print("[INFO] Calling get_page_content() ...")
        #             page_content = client.get_page_content(first_item["id"])
        #             print("[INFO] get_page_content() returned.")
        #             print("📄 Page Content Structure:")
        #             print(json.dumps(page_content, indent=2, default=str))
        #             print()
                    
        #             # Get block children
        #             print("📝 Getting block children...")
        #             print("[INFO] Calling get_block_children() ...")
        #             blocks = client.get_block_children(first_item["id"])
        #             print(f"[INFO] get_block_children() returned {len(blocks)} blocks.")
        #             print(f"Found {len(blocks)} blocks")
        #             for i, block in enumerate(blocks[:5]):  # Show first 5 blocks
        #                 print(f"  Block {i+1}: {block.get('type', 'unknown')} - {block.get('id', 'N/A')}")
        #             if len(blocks) > 5:
        #                 print(f"  ... and {len(blocks) - 5} more blocks")
        #             print()
                    
        #         except Exception as e:
        #             print(f"❌ Error getting page content: {e}")
        
        # # 4. If there are databases, get their content
        # databases = [item for item in child_pages if item["type"] == "child_database"]
        # if databases:
        #     print(f"🗃️  Found {len(databases)} databases. Getting content of first database...")
        #     first_db = databases[0]
        #     try:
        #         print("[INFO] Calling get_database_content() ...")
        #         db_pages = client.get_database_content(first_db["id"])
        #         print(f"[INFO] get_database_content() returned {len(db_pages)} pages.")
        #         print(f"Found {len(db_pages)} pages in database")
                
        #         for i, page in enumerate(db_pages[:3]):  # Show first 3 pages
        #             print(f"\n📄 Database Page {i+1}:")
        #             print_page_info(page, indent=1)
                
        #         if len(db_pages) > 3:
        #             print(f"  ... and {len(db_pages) - 3} more pages")
        #         print()
                
        #     except Exception as e:
        #         print(f"❌ Error getting database content: {e}")
        
        # # 5. Demonstrate search functionality
        # print("🔍 Demonstrating search functionality...")
        # try:
        #     print("[INFO] Calling search_pages('test') ...")
        #     search_results = client.search_pages("test")
        #     print(f"[INFO] search_pages() returned {len(search_results)} results.")
        #     print(f"Found {len(search_results)} pages matching 'test'")
        #     for i, result in enumerate(search_results[:3]):  # Show first 3 results
        #         print(f"\n🔍 Search Result {i+1}:")
        #         print_page_info(result, indent=1)
            
        #     if len(search_results) > 3:
        #         print(f"  ... and {len(search_results) - 3} more results")
        #     print()
            
        # except Exception as e:
        #     print(f"❌ Error searching pages: {e}")
        
        # print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("\nMake sure you have:")
        print("1. NOTION_API_KEY set in your .env file")
        print("2. NOTION_HOME_PAGE_ID set in your .env file")
        print("3. Your integration has access to the pages/databases")


if __name__ == "__main__":
    demo_fetch_pages() 