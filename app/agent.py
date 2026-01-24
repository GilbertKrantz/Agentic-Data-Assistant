from langchain.agents import create_agent
from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from app.config import settings


def agent_creator(
    tools: list[BaseTool],
    instruction: SystemMessage,
    model_name: str = "gemini-3-flash-preview",
):
    """Creates a LangChain agent with the provided tools and language model.

    Args:
        tools (list[BaseTool]): A list of LangChain Tool instances to be used by the agent.
        instruction: The system message/prompt for the agent.
        model_name: The Gemini model to use. Defaults to gemini-3-flash-preview.

    Returns:
        any: An instance of a LangChain agent configured with the provided tools and language model.
    """
    agent = create_agent(
        model=ChatGoogleGenerativeAI(model=model_name, api_key=settings.gemini_api_key),
        tools=tools,
        system_prompt=instruction,
    )

    return agent
