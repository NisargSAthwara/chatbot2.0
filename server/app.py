from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Initialize memory saver for checkpointing
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(
    max_results=4,
)

tools = [search_tool]

llm = ChatOpenAI( openai_api_key=os.getenv("TOGETHER_API_KEY"),
    openai_api_base="https://api.together.xyz/v1",
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    streaming=True)

llm_with_tools = llm.bind_tools(tools=tools)

async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages": [result], 
    }

async def tools_router(state: State):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END
    
async def tool_node(state):
    """Custom tool node that handles tool calls from the LLM."""
    # Get the tool calls from the last message
    tool_calls = state["messages"][-1].tool_calls
    
    # Initialize list to store tool messages
    tool_messages = []
    
    # Process each tool call
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Handle the search tool
        if tool_name == "tavily_search_results_json":
            # Execute the search tool with the provided arguments
            search_results = await search_tool.ainvoke(tool_args)
            
            # Create a ToolMessage for this result
            tool_message = ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            )
            
            tool_messages.append(tool_message)
    
    # Add the tool messages to the state
    return {"messages": tool_messages}

graph_builder = StateGraph(State)

graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")

graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)

app = FastAPI()

# Add CORS middleware with settings that match frontend requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatbot2-0-fwzq.onrender.com"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    expose_headers=["Content-Type"], 
)

def serialise_ai_message_chunk(chunk): 
    if(isinstance(chunk, AIMessageChunk)):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )

async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None
    
    if is_new_conversation:
        # Generate new checkpoint ID for first message in conversation
        new_checkpoint_id = str(uuid4())

        config = {
            "configurable": {
                "thread_id": new_checkpoint_id
            }
        }
        
        # Initialize with first message
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )
        
        # First send the checkpoint ID
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {
            "configurable": {
                "thread_id": checkpoint_id
            }
        }
        # Continue existing conversation
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

    async for event in events:
        event_type = event["event"]
        
        if event_type == "on_chat_model_stream":
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
            # Escape single quotes and newlines for safe JSON parsing
            safe_content = chunk_content.replace("'", "\\'").replace("\n", "\\n")
            
            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"
            
        elif event_type == "on_chat_model_end":
            # Check if there are tool calls for search
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "tavily_search_results_json"]
            
            if search_calls:
                # Signal that a search is starting
                search_query = search_calls[0]["args"].get("query", "")
                # Escape quotes and special characters
                safe_query = search_query.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
                yield f"data: {{\"type\": \"search_start\", \"query\": \"{safe_query}\"}}\n\n"
                
        elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
            # Search completed - send results or error
            output = event["data"]["output"]
            
            # Check if output is a list 
            if isinstance(output, list):
                # Extract URLs from list of search results
                urls = []
                for item in output:
                    if isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])
                
                # Convert URLs to JSON and yield them
                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\", \"urls\": {urls_json}}}\n\n"
    
    # Send an end event
    yield f"data: {{\"type\": \"end\"}}\n\n"

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), 
        media_type="text/event-stream"
    )

# SSE - server-sent events # import os 
# from typing import TypedDict, Annotated, Optional, List
# from langgraph.graph import add_messages , StateGraph, END
# from langchain.chat_models import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from dotenv import load_dotenv
# from langchain_community.tools.tavily_search import TavilySearchResults
# from fastapi import FastAPI, Query
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# import json
# from uuid import uuid4
# from langgraph.checkpoint.memory import MemorySaver

# load_dotenv()

# memory = MemorySaver()

# class Message(TypedDict):
#     messages: Annotated[List, add_messages]
# State = Message

# # Use a neutral description for the search tool
# search_tool = TavilySearchResults(
#     # max_results=4,
#     # description="Use this tool ONLY when the user explicitly asks for current, real-time, or up-to-date information from the web. Do NOT answer as if you have real-time access otherwise."
# )
# tools = [search_tool]

# llm = ChatOpenAI(
#     openai_api_key=os.getenv("TOGETHER_API_KEY"),
#     openai_api_base="https://api.together.xyz/v1",
#     model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
#     streaming=True
# )
# llm_with_tools = llm.bind(tools=tools)

# async def model(state : State):
#     result = await llm_with_tools.ainvoke(state["messages"])
#     return {
#         "messages": [result],
#     }

# async def tools_router(state : State):
#     last_message = state["messages"][-1]
#     print("Tool calls:", getattr(last_message, "tool_calls", None))  # <-- Place debug here

#     if (hasattr(last_message,"tool_calls") and len(last_message.tool_calls) > 0):
#         return "tool_node"
#     else:
#         return END

# async def tool_node(State):
#     """" Custom tool node that handles tool calls from the LLM. """
#     tool_calls = state["messages"][-1].tool_calls
#     tool_messages = []


#     for tool_call in tool_calls:
#         tool_name = tool_call["name"]
#         tool_args = tool_call["args"]
#         tool_id = tool_call["id"]

#         if tool_name == "tavily_search_results_json":
#             search_results = await search_tool.ainvoke(tool_args)

#             tool_message = ToolMessage(
#                 content = str(search_results),
#                 tool_call_id = tool_id,
#                 name = tool_name,
#             )

#             tool_messages.append(tool_message)

#     return {
#         "messages" : [tool_messages],
#     }

# graph_builder = StateGraph(State)
# graph_builder.add_node("model",model)
# graph_builder.add_node("tool_node",tool_node)
# graph_builder.set_entry_point("model")
# graph_builder.add_conditional_edges("model",tools_router)
# graph_builder.add_edge("tool_node","model")

# graph = graph_builder.compile(checkpointer=memory)
        
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],  
#     allow_headers=["*"], 
#     expose_headers=["Content-Type"], 
# )

# def serialise_ai_message_chunk(chunk):
#     if(isinstance(chunk,AIMessage)):
#         return chunk.content
#     else:
#         raise TypeError(f"Object of type {type(chunk).__name__} is not correctly formatted for serializable")

# # async def generate_chat_response(message:str , checkpoint_id: Optional[str] = None):
# #     is_new_conversation = checkpoint_id is None
# #     if is_new_conversation:
# #         # Generate new checkpoint ID for first message in conversation
# #         new_checkpoint_id = str(uuid4())

# #         config = {
# #             "configurable": {
# #                 "thread_id": new_checkpoint_id
# #             }
# #         }

# #         events =graph.astream_events(
# #                 {"messages" : [HumanMessage(content=message)]},
# #                 version = "v2",
# #                 config = config,
# #         )

# #         yield f"data: {json.dumps({'type': 'checkpoint', 'checkpoint_id': new_checkpoint_id})}\n\n"
# #     else:
# #         config = {
# #             "configurable" : {
# #                 "thread_id" : checkpoint_id
# #             }
# #         }

# #         events = graph.astream_events(
# #                  {"messages" : [HumanMessage(content=message)]},
# #                  version = "v2",
# #                  config = config,
# #         )


        
# #     async for event in events:
# #         event_type = event["event"]
# #         if event_type == "on_chat_model_stream":
# #             chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
# #             yield f"data: {json.dumps({'type': 'content', 'content': chunk_content})}\n\n"
# #         elif event_type == "on_chat_model_end":
# #             tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
# #             search_calls = [call for call in tool_calls if call["name"] == "tavily_search_results_json"]
# #             if search_calls:
# #                 search_query = search_calls[0]["args"].get("query", "")
# #                 yield f"data: {json.dumps({'type': 'search_start', 'query': search_query})}\n\n"
# #         elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
# #             output = event["data"]["output"]
# #             if isinstance(output, list):
# #                 urls = [item["url"] for item in output if isinstance(item, dict) and "url" in item]
# #                 yield f"data: {json.dumps({'type': 'search_results', 'urls': urls})}\n\n"
# #     yield f"data: {json.dumps({'type': 'end'})}\n\n"

# TAVILY_USAGE_LIMIT = 5
# tavily_usage_count = 0

# def is_real_time_query(query):
#     real_time_keywords = ["temperature", "stock price", "population", "date", "time"]
#     return any(keyword in query.lower() for keyword in real_time_keywords)

# async def generate_chat_response(message: str, checkpoint_id: Optional[str] = None):
#     global tavily_usage_count
#     is_new_conversation = checkpoint_id is None
#     if is_new_conversation:
#         # Generate new checkpoint ID for first message in conversation
#         new_checkpoint_id = str(uuid4())

#         config = {
#             "configurable": {
#                 "thread_id": new_checkpoint_id
#             }
#         }

#         events = graph.astream_events(
#             {"messages": [HumanMessage(content=message)]},
#             version="v2",
#             config=config,
#         )

#         yield f"data: {json.dumps({'type': 'checkpoint', 'checkpoint_id': new_checkpoint_id})}\n\n"
#     else:
#         config = {
#             "configurable": {
#                 "thread_id": checkpoint_id
#             }
#         }

#         events = graph.astream_events(
#             {"messages": [HumanMessage(content=message)]},
#             version="v2",
#             config=config,
#         )
     
#     if is_real_time_query(message) and tavily_usage_count < TAVILY_USAGE_LIMIT:
#         # Invoke Tavily search tool
#         search_results = await search_tool.ainvoke({"query": message})
#         tavily_usage_count += 1
#         yield f"data: {json.dumps({'type': 'search_start', 'query': message})}\n\n"
#     elif tavily_usage_count >= TAVILY_USAGE_LIMIT:
#         yield f"data: {json.dumps({'type': 'limit_reached', 'message': 'Tavily usage limit reached'})}\n\n"   

#     async for event in events:
#         event_type = event["event"]
#         if event_type == "on_chat_model_stream":
#             chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
#             yield f"data: {json.dumps({'type': 'content', 'content': chunk_content})}\n\n"
#         elif event_type == "on_chat_model_end":
#             tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
#             search_calls = [call for call in tool_calls if call["name"] == "tavily_search_results_json"]
#             if search_calls and tavily_usage_count < TAVILY_USAGE_LIMIT:
#                 search_query = search_calls[0]["args"].get("query", "")
#                 tavily_usage_count += 1
#                 yield f"data: {json.dumps({'type': 'search_start', 'query': search_query})}\n\n"
#             elif tavily_usage_count >= TAVILY_USAGE_LIMIT:
#                 yield f"data: {json.dumps({'type': 'limit_reached', 'message': 'Tavily usage limit reached'})}\n\n"
#         elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
#             output = event["data"]["output"]
#             if isinstance(output, list):
#                 urls = [item["url"] for item in output if isinstance(item, dict) and "url" in item]
#                 yield f"data: {json.dumps({'type': 'search_results', 'urls': urls})}\n\n"
#     yield f"data: {json.dumps({'type': 'end'})}\n\n"

# # Move this route definition OUTSIDE of the function above
# @app.get("/chat_stream/{messages}")
# async def chat_stream(messages: str, checkpoint_id: Optional[str] = Query(None)):
#     return StreamingResponse(
#         generate_chat_response(messages, checkpoint_id), media_type="text/event-stream")

# import os
# from typing import TypedDict, Annotated, Optional, List
# from langgraph.graph import add_messages, StateGraph, END
# from langchain.chat_models import ChatOpenAI
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# from langchain_community.tools.tavily_search import TavilySearchResults
# from dotenv import load_dotenv
# from fastapi import FastAPI, Query
# from fastapi.responses import StreamingResponse, PlainTextResponse
# from fastapi.middleware.cors import CORSMiddleware
# from uuid import uuid4
# from langgraph.checkpoint.memory import MemorySaver
# import json

# load_dotenv()

# memory = MemorySaver()

# class Message(TypedDict):
#     messages: Annotated[List, add_messages]

# State = Message

# search_tool = TavilySearchResults()
# tools = [search_tool]

# llm = ChatOpenAI(
#     openai_api_key=os.getenv("TOGETHER_API_KEY"),
#     openai_api_base="https://api.together.xyz/v1",
#     model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
#     streaming=True
# )

# llm_with_tools = llm.bind(tools=tools)

# async def model(state: State):
#     result = await llm_with_tools.ainvoke(state["messages"])
#     return {"messages": [result]}

# # async def tools_router(state: State):
# #     last_message = state["messages"][-1]
# #     if hasattr(last_message, "tool_calls") and last_message.tool_calls:
# #         return "tool_node"
# #     return END
# async def tools_router(state: State):
#     last_message = state["messages"][-1]
#     print("Last message:", last_message)
#     print("Has tool_calls attr:", hasattr(last_message, "tool_calls"))
#     if hasattr(last_message, "tool_calls"):
#         print("Tool calls:", last_message.tool_calls)
#     # Rest of the function

# async def tool_node(state: State):
#     tool_calls = state["messages"][-1].tool_calls
#     tool_messages = []

#     for tool_call in tool_calls:
#         tool_name = tool_call["name"]
#         tool_args = tool_call["args"]
#         tool_id = tool_call["id"]

#         if tool_name == "tavily_search_results_json":
#             search_results = await search_tool.ainvoke(tool_args)
#             tool_messages.append(
#                 ToolMessage(
#                     content=str(search_results),
#                     tool_call_id=tool_id,
#                     name=tool_name,
#                 )
#             )

#     return {"messages": tool_messages}

# graph_builder = StateGraph(State)
# graph_builder.add_node("model", model)
# graph_builder.add_node("tool_node", tool_node)
# graph_builder.set_entry_point("model")
# graph_builder.add_conditional_edges("model", tools_router)
# graph_builder.add_edge("tool_node", "model")

# graph = graph_builder.compile(checkpointer=memory)

# app = FastAPI()

# # ✅ CORS Middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def serialise_ai_message_chunk(chunk):
#     if isinstance(chunk, AIMessage):
#         return chunk.content
#     raise TypeError(f"Object of type {type(chunk).__name__} is not serializable")

# TAVILY_USAGE_LIMIT = 5
# tavily_usage_count = 0

# def is_real_time_query(query: str):
#     keywords = ["temperature", "weather", "date", "time", "news", "stock", "population"]
#     return any(k in query.lower() for k in keywords)

# async def generate_chat_response(message: str, checkpoint_id: Optional[str] = None):
#     global tavily_usage_count
#     is_new_conversation = checkpoint_id is None

#     if is_real_time_query(message) and tavily_usage_count < TAVILY_USAGE_LIMIT:
#         tavily_usage_count += 1
#         search_results = await search_tool.ainvoke({"query": message})
#         yield f"data: {json.dumps({'type': 'search_start', 'query': message})}\n\n"
#         urls = [item["url"] for item in search_results if isinstance(item, dict) and "url" in item]
#         yield f"data: {json.dumps({'type': 'search_results', 'urls': urls})}\n\n"

#     if is_new_conversation:
#         checkpoint_id = str(uuid4())
#         yield f"data: {json.dumps({'type': 'checkpoint', 'checkpoint_id': checkpoint_id})}\n\n"

#     config = {"configurable": {"thread_id": checkpoint_id}}

#     events = graph.astream_events(
#         {"messages": [HumanMessage(content=message)]},
#         version="v2",
#         config=config
#     )

#     async for event in events:
#         event_type = event["event"]

#         if event_type == "on_chat_model_stream":
#             chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
#             yield f"data: {json.dumps({'type': 'content', 'content': chunk_content})}\n\n"

#         elif event_type == "on_chat_model_end":
#             tool_calls = getattr(event["data"]["output"], "tool_calls", [])
#             if tool_calls and tavily_usage_count < TAVILY_USAGE_LIMIT:
#                 search_call = next((call for call in tool_calls if call["name"] == "tavily_search_results_json"), None)
#                 if search_call:
#                     search_query = search_call["args"].get("query", "")
#                     tavily_usage_count += 1
#                     yield f"data: {json.dumps({'type': 'search_start', 'query': search_query})}\n\n"
#                     # Print the usage count
#                     print(f"Tavily usage count: {tavily_usage_count}")
#             elif tavily_usage_count >= TAVILY_USAGE_LIMIT:
#                 yield f"data: {json.dumps({'type': 'limit_reached', 'message': 'Tavily usage limit reached'})}\n\n"

#         elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
#             output = event["data"]["output"]
#             urls = [item["url"] for item in output if isinstance(item, dict) and "url" in item]
#             yield f"data: {json.dumps({'type': 'search_results', 'urls': urls})}\n\n"

#     yield f"data: {json.dumps({'type': 'end'})}\n\n"

# @app.get("/chat_stream/{message}")
# async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
#     try:
#         return StreamingResponse(
#             generate_chat_response(message, checkpoint_id),
#             media_type="text/event-stream"
#         )
#     except Exception as e:
#         print("🚨 /chat_stream ERROR:", e)
#         return PlainTextResponse(str(e), status_code=500)