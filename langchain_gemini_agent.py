"""
[WRAPPER] Gemini-compatible tool agent creation

This module provides a replacement for LangChain's create_openai_tools_agent
that works with Gemini models using JSON-based tool calling.

[REAL API - GEMINI] Tool calling is simulated through JSON output and parsing,
then executed locally before being passed back to the LLM.

[COMPATIBILITY] Maintains interface compatibility with LangChain agents.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import re

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableConfig


class GeminiToolsAgent(Runnable):
    """
    [WRAPPER] Gemini-based tools agent
    
    This agent uses Gemini's JSON mode to decide which tools to call,
    rather than relying on OpenAI's native tool calling.
    
    [REAL API - GEMINI] The LLM makes real calls to Gemini API.
    [COMPATIBILITY] Implements Runnable protocol for LangChain compatibility.
    
    Flow:
    1. User provides input + tool descriptions
    2. Agent calls Gemini with JSON instructions
    3. Gemini returns JSON with tool decisions
    4. Parse JSON to extract tool calls
    5. Execute tools locally
    6. Pass results back to Gemini
    7. Gemini generates final response
    """
    
    def __init__(
        self,
        llm: Any,  # [WRAPPER] ChatGemini instance
        tools: Sequence[BaseTool],
        prompt: BasePromptTemplate,
        **kwargs: Any,
    ):
        """
        [WRAPPER] Initialize Gemini tools agent
        
        Args:
            llm: Language model (should be ChatGemini or compatible)
            tools: List of tools the agent can use
            prompt: Prompt template for the agent
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.prompt = prompt
        self.tool_names = list(self.tools.keys())
        
        # [WRAPPER] Create tool descriptions for prompting
        self.tool_descriptions = self._create_tool_descriptions()
    
    def _create_tool_descriptions(self) -> str:
        """
        [WRAPPER] Create JSON schema for tools
        
        Returns description of available tools in a format Gemini can understand.
        """
        tools_desc = {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "description": "Tool input"}
                        },
                        "required": ["input"],
                    },
                }
                for tool in self.tools.values()
            ]
        }
        return json.dumps(tools_desc, indent=2)
    
    def _parse_tool_calls(self, response_text: str) -> List[Tuple[str, str]]:
        """
        [WRAPPER] Parse JSON tool calls from LLM response
        
        Attempts to extract JSON tool calls from the LLM response.
        Looks for patterns like:
        ```json
        {"tool": "tool_name", "input": "tool_input"}
        ```
        
        Args:
            response_text: Raw response from LLM
        
        Returns:
            List of (tool_name, tool_input) tuples
        """
        tool_calls = []
        
        # [WRAPPER] Try to find JSON blocks in response
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                tool_call = json.loads(match)
                if "tool" in tool_call and "input" in tool_call:
                    tool_calls.append((tool_call["tool"], tool_call["input"]))
            except json.JSONDecodeError:
                pass
        
        # [WRAPPER] Fallback: try direct JSON parsing
        if not tool_calls:
            try:
                tool_call = json.loads(response_text)
                if "tool" in tool_call and "input" in tool_call:
                    tool_calls.append((tool_call["tool"], tool_call["input"]))
            except json.JSONDecodeError:
                pass
        
        return tool_calls
    
    def invoke(
        self,
        input: Union[Dict, str, BaseMessage],
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        """
        [WRAPPER] Run one step of the agent
        
        Args:
            input: Agent input
            config: Configuration
        
        Returns:
            Agent action or finish signal
        """
        # [WRAPPER] Convert input to message format
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        elif isinstance(input, dict):
            messages = [HumanMessage(content=str(input))]
        else:
            messages = [input] if isinstance(input, list) else [input]
        
        # [REAL API - GEMINI] Get LLM response with tool descriptions
        prompt_text = f"""
Available tools:
{self.tool_descriptions}

Respond with JSON indicating which tool to use, in this format:
{{"tool": "tool_name", "input": "tool_input"}}

Or provide your final answer if no tool is needed.
"""
        
        # [REAL API - GEMINI] Call LLM
        messages_with_tools = messages + [HumanMessage(content=prompt_text)]
        response = self.llm.invoke(messages_with_tools)
        
        # [WRAPPER] Parse tool calls from response
        tool_calls = self._parse_tool_calls(response.content)
        
        if tool_calls:
            tool_name, tool_input = tool_calls[0]
            
            # [WRAPPER] Check if tool exists
            if tool_name in self.tools:
                return AgentAction(
                    tool=tool_name,
                    tool_input=tool_input,
                    log=response.content,
                )
        
        # [WRAPPER] No tool call, return final response
        return AgentFinish(
            output=response.content,
            log=response.content,
        )
    
    def stream(self, *args, **kwargs):
        """[WRAPPER] Streaming not yet supported for Gemini tools agent"""
        return self.invoke(*args, **kwargs)


def create_gemini_tools_agent(
    llm: Any,  # [WRAPPER] ChatGemini instance
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    **kwargs: Any,
) -> Runnable:
    """
    [WRAPPER] Create a Gemini-based tools agent
    
    This is a drop-in replacement for LangChain's create_openai_tools_agent
    but uses Gemini instead of OpenAI.
    
    [REAL API - GEMINI] The created agent will use Gemini for decision making.
    [COMPATIBILITY] Returns a Runnable compatible with LangChain's AgentExecutor.
    
    Args:
        llm: Language model (should be ChatGemini)
        tools: List of tools the agent can use
        prompt: Prompt template
        **kwargs: Additional arguments
    
    Returns:
        Agent runnable compatible with AgentExecutor
    
    Example:
        from langchain_gemini_wrapper import ChatGemini
        from langchain_gemini_agent import create_gemini_tools_agent
        from langchain.agents import AgentExecutor
        
        llm = ChatGemini(api_key="your_key")
        tools = [your_tools]
        prompt = your_prompt
        
        agent = create_gemini_tools_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        
        result = executor.invoke({"input": "your question"})
    """
    return GeminiToolsAgent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        **kwargs,
    )
