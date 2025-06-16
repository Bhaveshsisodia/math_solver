import gradio as gr
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
import time

class GradioCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs = []

    def on_tool_start(self, tool, input_str, **kwargs):
        msg = f"🔧 **Using tool `{tool.name}` with input**: `{input_str}`"
        self.logs.append(msg)
        yield msg

    def on_tool_end(self, output, **kwargs):
        msg = f"✅ **Tool Output**: {output}"
        self.logs.append(msg)
        yield msg

    def on_text(self, text, **kwargs):
        msg = f"📝 {text.strip()}"
        self.logs.append(msg)
        yield msg

def initialize_agent_with_key_streaming(groq_api_key):
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

    wikipedia_wrapper = WikipediaAPIWrapper()
    wikipedia_tool = Tool(
        name="Wikipedia",
        func=wikipedia_wrapper.run,
        description="Wikipedia Search Tool"
    )

    math_chain = LLMMathChain.from_llm(llm=llm)
    calculator = Tool(
        name="Calculator",
        func=math_chain.run,
        description="Mathematical calculator tool"
    )

    prompt = """
    You are an agent tasked with solving users' mathematical questions. Provide a detailed step-wise explanation.

    Question: {question}
    Answer:
    """
    prompt_template = PromptTemplate(input_variables=["question"], template=prompt)
    reasoning_chain = LLMChain(llm=llm, prompt=prompt_template)

    reasoning_tool = Tool(
        name="Reasoning Tool",
        func=reasoning_chain.run,
        description="Logical reasoning tool"
    )

    assistant_agent = initialize_agent(
        tools=[wikipedia_tool, calculator, reasoning_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    return assistant_agent

def gradio_stream(groq_api_key, question):
    if not groq_api_key:
        yield "❗ Please enter your Groq API key."
        return

    assistant_agent = initialize_agent_with_key_streaming(groq_api_key)
    callback = GradioCallbackHandler()

    # Run the agent while streaming intermediate outputs
    try:
        yield "🚀 Starting processing..."
        response = assistant_agent.run(question, callbacks=[callback])
        for log in callback.logs:
            yield log
        yield f"🎯 **Final Answer**: {response}"
    except Exception as e:
        yield f"❗ Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🧮 Text to Math Problem Solver with Streaming Output")
    groq_api_key = gr.Textbox(label="Groq API Key", type="password")
    question = gr.Textbox(label="Your Question")

    output = gr.Textbox(label="Output", lines=15)

    generate_btn = gr.Button("Submit (Stream Output)")

    generate_btn.click(fn=gradio_stream, inputs=[groq_api_key, question], outputs=output, api_name="stream", queue=True)

demo.launch()
