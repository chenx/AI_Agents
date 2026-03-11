from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_classic.agents import AgentExecutor
from langchain_classic import hub
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent
from langchain_classic.memory import ConversationBufferMemory

import requests
from ddgs import DDGS


# https://www.meteomatics.com/en/weather-api/how-to-get-weather-api-key/
# For business use only.
class WeatherTool(BaseTool):
    name: str = "WeatherAPI"
    description: str = "Get current weather for a city"

    def _run(self, city: str) -> str:
        if not city:
            return "No input provided, stopping execution."
        url = f"https://api.weatherapi.com/v1/current.json?key=API_KEY&q={city}"
        response = requests.get(url)
        return response.json()


class EventTool(BaseTool):
    name: str = "EventAPI"
    description: str = "Get events"

    def _run(self, event: str) -> str:
        # TODO: need to setup this.
        url = f"https://*.ngrok-free.dev/api/events/"
        response = requests.get(url)
        return response.json()


class SearchTool(BaseTool):
    name: str = "WebSearch"
    description: str = "Search the web for recent information"

    def _run(self, query: str) -> str:
        if not query:
            return "No input provided, stopping execution."
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            return [r["title"] for r in results]


class NoOpTool(BaseTool):
    name: str = "noop"
    description: str = "Fallback tool when LLM output is invalid"

    def _run(self, query: str) -> str:
        return "Final Answer: Stopped because no valid action was parsed."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError()


llm = ChatOpenAI(
    # model="gpt-4",
    model="gpt-4o-mini",
    temperature=0
)

# tools = [SearchTool(), EventTool(), NoOpTool()]
tools = [SearchTool(), NoOpTool()]

memory = ConversationBufferMemory(memory_key="chat_history")

# https://smith.langchain.com/hub/hwchase17/react
# prompt = hub.pull("hwchase17/react")

# Create a proper ReAct prompt
# Must include {input} and {agent_scratchpad}
react_prompt_template = """
You are a ReAct agent. Follow this format:

Thought: <what you are thinking>
Action: <tool name>
Action Input: <tool input>
Observation: <result from tool>

If you cannot proceed, stop and give:
Final Answer: <your answer>

Available tools:
{tools}
Tool names: {tool_names}

Question: {input}

{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(react_prompt_template)

agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=prompt
)
# agent.run("Find the latest AI news and summarize it.")


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    stop_on_invalid_action=True,
)
# agent_executor.invoke({"input": "Find the latest AI news and summarize it"})
# agent_executor.invoke({"input": "Get events of Jackie Chan"})
agent_executor.invoke({"input": "Who is obama?"})
