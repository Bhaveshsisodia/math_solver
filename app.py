import gradio as gr
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType


#bhavesh
# Initialize global variables
llm = None
assistant_agent = None

def initialize_agent_with_key(groq_api_key):
    global llm, assistant_agent

    # Initialize LLM
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

    # Wikipedia Tool
    wikipedia_wrapper = WikipediaAPIWrapper()
    wikipedia_tool = Tool(
        name="Wikipedia",
        func=wikipedia_wrapper.run,
        description="A tool for searching the Internet to find the various information on the topics mentioned",
    )

    # Math Tool
    math_chain = LLMMathChain.from_llm(llm=llm)
    calculator = Tool(
        name="Calculator",
        func=math_chain.run,
        description="A tool for answering math related questions. Only input mathematical expressions need to be provided",
    )

    # Reasoning Tool
    prompt = """
    You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed explanation
    and display it point-wise for the question below:
    
    Question: {question}
    Answer:
    """
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=prompt
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    reasoning_tool = Tool(
        name="Reasoning tool",
        func=chain.run,
        description="A tool for answering logic-based and reasoning questions.",
    )

    # Initialize the Agent
    assistant_agent = initialize_agent(
        tools=[wikipedia_tool, calculator, reasoning_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )

def process_question(groq_api_key, question):
    if not groq_api_key:
        return "❗ Please enter your Groq API key."

    if not assistant_agent or not llm:
        initialize_agent_with_key(groq_api_key)

    try:
        response = assistant_agent.run(question)
        return response
    except Exception as e:
        return f"❗ Error: {str(e)}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧮 Text to Math Problem Solver with Wikipedia Search (Gemma2-9b-It)")
    gr.Markdown("Solve mathematical problems with step-by-step reasoning and search information from Wikipedia.")

    groq_api_key = gr.Textbox(label="🔑 Groq API Key (Required)", type="password")
    question = gr.Textbox(label="💬 Enter Your Question")

    output = gr.Markdown()

    submit_btn = gr.Button("Generate Response")

    submit_btn.click(process_question, inputs=[groq_api_key, question], outputs=output)

demo.launch()
