"""Tool prompt templates for the email assistant."""

# Tool descriptions for agent workflow without triage
AGENT_TOOLS_PROMPT = """
1. write_email(to, subject, content) - Send emails to specified recipients
4. Done - E-mail has been sent
"""