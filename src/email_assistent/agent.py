from dotenv import load_dotenv
import os

load_dotenv()

from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


from tools.base import get_tools, get_tools_by_name
from tools.default.prompt_templates import AGENT_TOOLS_PROMPT
from prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt, default_background, default_triage_instructions, default_response_preferences, default_cal_preferences
from schemas import State, RouterSchema, StateInput
from utils import parse_gmail, show_graph, parse_email, format_email_markdown, show_graph_terminal


tools = get_tools()
tools_by_name = get_tools_by_name(tools)

# LLM for classificaion of incomming email
llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
llm_router = llm.with_structured_output(RouterSchema) 

#LLM with bind tool
llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
llm_with_tools = llm.bind_tools(tools, tool_choice="any")

# Nodes
def llm_call(state: State):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(content= agent_system_prompt.format(
                        tools_prompt=AGENT_TOOLS_PROMPT,
                        background=default_background,
                        response_preferences=default_response_preferences, 
                        cal_preferences=default_cal_preferences)
                    )
                    
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state: State):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content" : observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

# Conditional edge function
def should_continue(state: State) -> Literal["Action", "__end__"]:
    """Route to Action, or end if Done tool called"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls: 
            if tool_call["name"] == "Done":
                return END
            else:
                return "Action"
            
# Build workflow
agent_builder = StateGraph(State)

#Add_Nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """
    

    author, to, subject, email_thread = parse_email(state["email_input"])

    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
    )
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('_++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(1)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('_++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(2)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    result=llm_router.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]    
    )

    # result = llm_router.invoke(
    #     [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ]
    # )

    # input=[
    #         SystemMessage(content=system_prompt),
    #         HumanMessage(content=user_prompt)
    #     ]  

    # for chunk in llm_router.stream(input):
    #     for m in chunk["messages"]:
    #         m.pretty_print()

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('_++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(result.classification)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    # Decision
    classification = result.classification

    if classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "classification_decision": result.classification,
            "messages": [
                HumanMessage(content=f"Respond to the email: {email_markdown}")
            ],
        }

    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update =  {
            "classification_decision": result.classification,
        }
        goto = END
    elif result.classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = {
            "classification_decision": result.classification,
        }
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)


full_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router")
)

email_assistant=full_workflow.compile()

email_input = {
    "author": "System Admin <sysadmin@company.com>",
    "to": "Development Team <dev@company.com>",
    "subject": "Scheduled maintenance - database downtime",
    "email_thread": "Hi team,\n\nThis is a reminder that we'll be performing scheduled maintenance on the production database tonight from 2AM to 4AM EST. During this time, all database services will be unavailable.\n\nPlease plan your work accordingly and ensure no critical deployments are scheduled during this window.\n\nThanks,\nSystem Admin Team"
}

response = email_assistant.invoke({"email_input": email_input})
for m in response["messages"]:
    m.pretty_print()

# show_graph_terminal(email_assistant)

