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
            text = f'__{text}__'  # Markdown doesn't support underline natively
        md += text
    return md

# Universal dict for numbered_counters
universal_numbered_counters = {}

def process_blocks(blocks, client, indent_level=0, numbered_counters=None, last_was_numbered=None):
    if numbered_counters is None:
        numbered_counters = universal_numbered_counters
    if last_was_numbered is None:
        last_was_numbered = {}
    for block in blocks:
        display_block_content(block, client, indent_level, numbered_counters, last_was_numbered)
        # If the block is not a numbered_list_item, reset the counter for this level
        if block.get("type") != "numbered_list_item" and block.get("type") != "child_page":
            if indent_level in numbered_counters:
                del numbered_counters[indent_level]
            if indent_level in last_was_numbered:
                del last_was_numbered[indent_level]

def display_block_content(block, client, indent_level=0, numbered_counters=None, last_was_numbered=None):
    """
    Recursively display block content, handling nested toggles and numbered lists.
    Top-level numbered_list_items count up (1, 2, 3, ...), and sub-lists at each indentation level start from 1, but parent numbering continues for siblings. Numbering resets when a non-numbered_list_item is encountered at that level.
    """
    if numbered_counters is None:
        numbered_counters = universal_numbered_counters
    if last_was_numbered is None:
        last_was_numbered = {}
    indent = "  " * indent_level

    # Numbered list item handling
    if block.get("type") == "numbered_list_item":
        # Create counter for this level if not present
        if indent_level not in numbered_counters:
            numbered_counters[indent_level] = 1
        counter = numbered_counters[indent_level]
        numbered_counters[indent_level] += 1
        last_was_numbered[indent_level] = True
        block_type = block["type"]
        rich_text_list = block.get(block_type, {}).get("rich_text", [])
        md = format_rich_text_to_markdown(rich_text_list) if rich_text_list else ""
        print(f"{indent}{counter}. {md}")
        # Handle children (sub-blocks, e.g. nested numbered lists)
        if block.get("has_children"):
            try:
                children = client.get_block_children(block["id"])
                # If children contain numbered_list_items, ensure a new counter for the child level
                has_numbered = any(child.get("type") == "numbered_list_item" for child in children)
                if has_numbered:
                    numbered_counters[indent_level + 1] = 1
                    last_was_numbered[indent_level + 1] = False
                process_blocks(children, client, indent_level + 1, numbered_counters, last_was_numbered)
            except Exception as e:
                print(f"{indent}  ❌ Error loading numbered list children: {e}")
        return
    # Reset both the counter and numbered status for this level
    if indent_level in numbered_counters:
        del numbered_counters[indent_level]
    last_was_numbered[indent_level] = False

    # Toggle block handling
    if block.get("type") == "toggle":
        toggle_title = format_rich_text_to_markdown(block["toggle"]["rich_text"])
        print(f"{indent}📋 {toggle_title}")
        if block.get("has_children"):
            try:
                toggle_children = client.get_block_children(block["id"])
                process_blocks(toggle_children, client, indent_level + 1, numbered_counters, last_was_numbered)
            except Exception as e:
                print(f"{indent}  ❌ Error loading toggle content: {e}")
        return

    # Regular block handling
    if block.get("type") != "child_page":
        block_type = block["type"]
        rich_text_list = block.get(block_type, {}).get("rich_text", [])
        if rich_text_list:
            md = format_rich_text_to_markdown(rich_text_list)
            print(f"{indent}• {md}")
        # If this block has children, recurse
        if block.get("has_children"):
            try:
                children = client.get_block_children(block["id"])
                process_blocks(children, client, indent_level + 1, numbered_counters, last_was_numbered)
            except Exception as e:
                print(f"{indent}  ❌ Error loading block children: {e}")


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


def interactive_page_browser(client):
    """
    Interactive page browser that allows users to navigate through Notion pages
    and select which pages to view.
    """
    # Stack to keep track of navigation history
    page_stack = []
    current_page_id = client.home_page_id
    
    while True:
        # Get current page metadata and blocks
        try:
            metadata = client.get_page_content(current_page_id)
            blocks = client.get_block_children(current_page_id)
        except Exception as e:
            print(f"❌ Error loading page: {e}")
            if page_stack:
                current_page_id = page_stack.pop()
                continue
            else:
                print("❌ Cannot recover from error. Exiting.")
                break
        
        # Get page title
        title = metadata["properties"]["title"]["title"][0]["text"]["content"] if metadata["properties"]["title"]["title"] else "Untitled"
        
        # Clear screen and show current page info
        print("\n" + "="*60)
        print(f"📄 {title}")
        print("="*60)
        
        # Extract and display page content
        print("\n📝 Page Content:")
        content_found = False
        for block in blocks:
            if block.get("type") != "child_page":
                display_block_content(block, client)
                content_found = True
        
        if not content_found:
            print("(No content found)")
        
        # Find sub-pages
        sub_pages = []
        for block in blocks:
            if block.get("type") == "child_page":
                sub_pages.append({
                    "id": block["id"],
                    "title": block["child_page"].get("title", "Untitled")
                })
        
        # Show navigation options
        print(f"\n📂 Available Sub-pages ({len(sub_pages)}):")
        if sub_pages:
            for i, page in enumerate(sub_pages, 1):
                print(f"  {i}. {page['title']}")
        else:
            print("  (No sub-pages found)")
        
        # Show navigation commands
        print(f"\n🔧 Navigation Commands:")
        print(f"  • Enter a number (1-{len(sub_pages)}) to navigate to that sub-page")
        print(f"  • 'back' to go to the previous page")
        print(f"  • 'home' to go to the home page")
        print(f"  • 'quit' or 'exit' to exit")
        
        # Get user input
        try:
            choice = input(f"\n🎯 Your choice: ").strip().lower()
            
            if choice in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif choice == 'back':
                if page_stack:
                    current_page_id = page_stack.pop()
                    print(f"⬅️ Going back...")
                else:
                    print("❌ Already at the root page!")
            
            elif choice == 'home':
                if current_page_id != client.home_page_id:
                    page_stack = []  # Clear history when going home
                    current_page_id = client.home_page_id
                    print(f"🏠 Going to home page...")
                else:
                    print("❌ Already at the home page!")
            
            elif choice.isdigit():
                page_num = int(choice)
                if 1 <= page_num <= len(sub_pages):
                    selected_page = sub_pages[page_num - 1]
                    page_stack.append(current_page_id)  # Save current page to history
                    current_page_id = selected_page["id"]
                    print(f"➡️ Navigating to: {selected_page['title']}")
                else:
                    print(f"❌ Invalid page number. Please enter 1-{len(sub_pages)}")
            
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing input: {e}")


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
        # blocks = client.get_block_children(client.home_page_id)
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

        # # Call the new markdown extraction function
        # print("\n================ MARKDOWN EXTRACTION ================")
        # extract_and_print_markdown_from_page(client.home_page_id, client)
        # print("====================================================\n")

        # Start interactive browser
        print("\n================ INTERACTIVE PAGE BROWSER ================")
        interactive_page_browser(client)
        print("==========================================================\n")

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