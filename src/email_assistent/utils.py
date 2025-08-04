def format_email_markdown(subject, author, to, email_thread, email_id=None):
    """Format email details into a nicely formatted markdown string for display
    
    Args:
        subject: Email subject
        author: Email sender
        to: Email recipient
        email_thread: Email content
        email_id: Optional email ID (for Gmail API)
    """
    id_section = f"\n**ID**: {email_id}" if email_id else ""
    
    return f"""

**Subject**: {subject}
**From**: {author}
**To**: {to}{id_section}

{email_thread}

---
"""

def parse_email(email_input: dict) -> dict:
    """Parse an email input dictionary.

    Args:
        email_input (dict): Dictionary containing email fields:
            - author: Sender's name and email
            - to: Recipient's name and email
            - subject: Email subject line
            - email_thread: Full email content

    Returns:
        tuple[str, str, str, str]: Tuple containing:
            - author: Sender's name and email
            - to: Recipient's name and email
            - subject: Email subject line
            - email_thread: Full email content
    """
    return (
        email_input["author"],
        email_input["to"],
        email_input["subject"],
        email_input["email_thread"],
    )



def parse_gmail(email_input: dict) -> tuple[str, str, str, str, str]:
    """Parse an email input dictionary for Gmail, including the email ID.
    
    This function extends parse_email by also returning the email ID,
    which is used specifically in the Gmail integration.

    Args:
        email_input (dict): Dictionary containing email fields in any of these formats:
            Gmail schema:
                - From: Sender's email
                - To: Recipient's email
                - Subject: Email subject line
                - Body: Full email content
                - Id: Gmail message ID
            
    Returns:
        tuple[str, str, str, str, str]: Tuple containing:
            - author: Sender's name and email
            - to: Recipient's name and email
            - subject: Email subject line
            - email_thread: Full email content
            - email_id: Email ID (or None if not available)
    """

    print("!Email_input from Gmail!")
    print(email_input)

    # Gmail schema
    return (
        email_input["from"],
        email_input["to"],
        email_input["subject"],
        email_input["body"],
        email_input["id"],
    )



def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with fallback rendering.
    
    Handles timeout errors from mermaid.ink by falling back to pyppeteer.
    
    Args:
        graph: The LangGraph object that has a get_graph() method
    """
    from IPython.display import Image
    try:
        # Try the default renderer first
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        # Fall back to pyppeteer if the default renderer fails
        import nest_asyncio
        nest_asyncio.apply()
        from langchain_core.runnables.graph import MermaidDrawMethod
        return Image(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER))
    

def show_graph_terminal(graph, xray=False, output_file="graph.png"):
    """Generate and save a LangGraph mermaid diagram as an image file for terminal use."""
    try:
        # Try default renderer first
        img_bytes = graph.get_graph(xray=xray).draw_mermaid_png()
    except Exception:
        # Fall back to pyppeteer if needed
        import nest_asyncio
        nest_asyncio.apply()
        from langchain_core.runnables.graph import MermaidDrawMethod
        img_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
    
    # Save image to file
    with open(output_file, "wb") as f:
        f.write(img_bytes)
    print(f"Graph saved to {output_file}")
    