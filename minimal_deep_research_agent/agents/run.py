from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union, cast

import openai
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessage, ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

from .agent import Agent
from .model_settings import ModelSettings

# Define minimal classes needed
class NextStepFinalOutput:
    def __init__(self, output):
        self.output = output

class NextStepHandoff:
    def __init__(self, new_agent):
        self.new_agent = new_agent

class NextStepRunAgain:
    pass

class QueueCompleteSentinel:
    pass

class SingleStepResult:
    def __init__(self, model_response, original_input, generated_items, next_step):
        self.model_response = model_response
        self.original_input = original_input
        self.generated_items = generated_items
        self.next_step = next_step

T = TypeVar('T')

@dataclass
class ModelResponse:
    output: Any
    usage: Dict[str, Any] = field(default_factory=dict)
    referenceable_id: str = ""

@dataclass
class RunResult:
    input: Any
    new_items: List = field(default_factory=list)
    raw_responses: List = field(default_factory=list)
    final_output: Any = None
    _last_agent: Any = None
    input_guardrail_results: List = field(default_factory=list)
    output_guardrail_results: List = field(default_factory=list)
    
    def final_output_as(self, cls):
        """Convert the final output to the given class"""
        if self.final_output is None:
            return None
        
        if isinstance(self.final_output, cls):
            return self.final_output
        
        # If final_output is a dict and cls is a BaseModel (pydantic)
        if isinstance(self.final_output, dict) and issubclass(cls, BaseModel):
            return cls(**self.final_output)
        
        # For simple string conversion
        if cls is str:
            return str(self.final_output)
        
        return self.final_output

@dataclass
class RunResultStreaming:
    input: Any
    new_items: List = field(default_factory=list)
    raw_responses: List = field(default_factory=list)
    final_output: Any = None
    current_agent: Any = None
    is_complete: bool = False
    current_turn: int = 0
    max_turns: int = 10
    input_guardrail_results: List = field(default_factory=list)
    output_guardrail_results: List = field(default_factory=list)
    _current_agent_output_schema: Any = None
    _trace: Any = None
    _event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _input_guardrail_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _input_guardrails_task: asyncio.Task = None
    _output_guardrails_task: asyncio.Task = None
    _run_impl_task: asyncio.Task = None
    
    async def stream_events(self):
        """Stream events from the agent run"""
        while True:
            event = await self._event_queue.get()
            if event == "COMPLETE":
                break
            yield event
            
    def final_output_as(self, cls):
        """Convert the final output to the given class"""
        if self.final_output is None:
            return None
        
        if isinstance(self.final_output, cls):
            return self.final_output
        
        # If final_output is a dict and cls is a BaseModel (pydantic)
        if isinstance(self.final_output, dict) and issubclass(cls, BaseModel):
            return cls(**self.final_output)
        
        # For simple string conversion
        if cls is str:
            return str(self.final_output)
        
        return self.final_output

class Runner:
    """Runs agents with the OpenAI API"""
    
    @classmethod
    async def run(
        cls,
        starting_agent: Agent[T],
        input: str,
        *,
        context: Optional[T] = None,
        max_turns: int = 10,
        hooks: Any = None,
        run_config: Any = None,
    ) -> RunResult:
        """Run an agent with the given input"""
        client = openai.AsyncOpenAI()
        
        # Transform input to messages
        messages = []
        
        # Add system message if available
        system_prompt = await starting_agent.get_system_prompt(context)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": input})
        
        # Prepare tools
        tools = [tool.as_dict() for tool in starting_agent.tools]
        
        # Get model
        model = starting_agent.model or "gpt-4o"
        
        # Get model settings
        model_settings = starting_agent.model_settings
        
        # Add output formatting instructions if there's an output_type
        if starting_agent.output_type and starting_agent.output_type is not str:
            if hasattr(starting_agent.output_type, '__annotations__'):
                # Get fields from pydantic model
                fields = starting_agent.output_type.__annotations__.keys()
                fields_str = ", ".join(fields)
                
                # Add this to the system prompt if not already there
                formatting_instruction = (
                    f"\nYou must format your response as a valid JSON object with these fields: {fields_str}. "
                    f"Make sure to include all required fields and properly format nested objects."
                )
                
                # Update system prompt if it exists
                if messages and messages[0]["role"] == "system":
                    if formatting_instruction not in messages[0]["content"]:
                        messages[0]["content"] += formatting_instruction
                # Add system prompt if it doesn't exist
                else:
                    messages.insert(0, {"role": "system", "content": formatting_instruction})
        
        # Call the API
        response = await client.chat.completions.create(
            model=model,
            messages=cast(List[ChatCompletionMessageParam], messages),
            tools=cast(List[ChatCompletionToolParam], tools) if tools else None,
            temperature=model_settings.temperature or 0.7,
            top_p=model_settings.top_p or 1.0,
            presence_penalty=model_settings.presence_penalty or 0.0,
            frequency_penalty=model_settings.frequency_penalty or 0.0,
            tool_choice="auto" if tools else None,
            response_format={"type": "json_object"} if starting_agent.output_type and starting_agent.output_type is not str else None,
        )
        
        # Process response
        assistant_message = response.choices[0].message
        
        # Check if there are tool calls
        if assistant_message.tool_calls:
            # Here we would execute the tool calls, but for simplicity
            # we'll just return a dummy result
            all_results = []
            for tool_call in assistant_message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    # In a real implementation, we would execute the tool
                    # For now, just return a dummy result
                    result = f"Result for {tool_call.function.name} with arguments {arguments}"
                    all_results.append(result)
                except Exception as e:
                    all_results.append(f"Error executing tool: {str(e)}")
            
            # Combine all results
            final_output = "\\n".join(all_results)
        else:
            # If no tool calls, just return the content
            final_output = assistant_message.content
        
        # If there's an output_type, try to convert the output
        if starting_agent.output_type and starting_agent.output_type is not str:
            if issubclass(starting_agent.output_type, BaseModel):
                try:
                    # Try to parse the output as JSON
                    if isinstance(final_output, str):
                        # Try multiple JSON extraction methods
                        json_data = None
                        
                        # Method 1: Direct JSON parsing if it's clean JSON
                        try:
                            json_data = json.loads(final_output)
                        except json.JSONDecodeError:
                            pass
                            
                        # Method 2: Look for JSON between curly braces
                        if json_data is None:
                            import re
                            json_match = re.search(r'({[\s\S]*})', final_output)
                            if json_match:
                                try:
                                    json_data = json.loads(json_match.group(1))
                                except json.JSONDecodeError:
                                    pass
                                    
                        # Method 3: Try to extract a JSON object with a more lenient regex
                        if json_data is None:
                            json_match = re.search(r'({[^{}]*({[^{}]*})*[^{}]*})', final_output)
                            if json_match:
                                try:
                                    json_data = json.loads(json_match.group(1))
                                except json.JSONDecodeError:
                                    pass
                                    
                        # If we found JSON data, try to create the output type
                        if json_data:
                            try:
                                final_output = starting_agent.output_type(**json_data)
                            except Exception:
                                # Silently continue with original output if parsing fails
                                pass
                        # If no JSON data found, continue with original output
                            
                except Exception as e:
                    print(f"Error converting output to {starting_agent.output_type.__name__}: {str(e)}")
                    # If conversion fails, leave as is
                    pass
        
        # Create the model response
        model_response = ModelResponse(
            output=assistant_message.content,
            usage={"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens},
            referenceable_id=response.id
        )
        
        # Create and return the result
        return RunResult(
            input=input,
            new_items=[],
            raw_responses=[model_response],
            final_output=final_output,
            _last_agent=starting_agent
        )
    
    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[T],
        input: str,
        *,
        context: Optional[T] = None,
        max_turns: int = 10,
        hooks: Any = None,
        run_config: Any = None,
    ) -> RunResult:
        """Synchronous version of run"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            cls.run(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
            )
        )
    
    @classmethod
    def run_streamed(
        cls,
        starting_agent: Agent[T],
        input: str,
        context: Optional[T] = None,
        max_turns: int = 10,
        hooks: Any = None,
        run_config: Any = None,
    ) -> RunResultStreaming:
        """Run an agent in streaming mode"""
        # Create a streamed result
        streamed_result = RunResultStreaming(
            input=input,
            current_agent=starting_agent,
            max_turns=max_turns,
        )
        
        # Start running the agent in the background
        streamed_result._run_impl_task = asyncio.create_task(
            cls._run_streamed_impl(streamed_result, starting_agent, input)
        )
        
        return streamed_result
    
    @classmethod
    async def _run_streamed_impl(
        cls,
        streamed_result: RunResultStreaming,
        agent: Agent[T],
        input: str,
    ):
        """Implementation of streaming mode"""
        try:
            # Run the agent normally
            result = await cls.run(agent, input)
            
            # Set the result values on the streamed result
            streamed_result.final_output = result.final_output
            streamed_result.raw_responses = result.raw_responses
            streamed_result.new_items = result.new_items
            streamed_result.is_complete = True
            
            # Signal completion
            streamed_result._event_queue.put_nowait("COMPLETE")
        except Exception as e:
            # Signal completion with error
            streamed_result._event_queue.put_nowait("COMPLETE")
            raise e

# Compatibility functions and variables
def gen_trace_id():
    """Generate a unique trace ID"""
    import uuid
    return str(uuid.uuid4())

class custom_span:
    """Context manager for custom spans"""
    def __init__(self, name: str):
        self.name = name
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class trace:
    """Context manager for traces"""
    def __init__(self, name: str, trace_id: str = None):
        self.name = name
        self.trace_id = trace_id or gen_trace_id()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass