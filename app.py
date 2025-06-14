import gradio as gr
from typing import List
from pydantic import BaseModel

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from src.utils import read_pdf, get_llm
from src.tools import chunk_embedding_tool, matching_policy_tool, similar_document_tool

# Define structured output format
class PolicyComplianceResponse(BaseModel):
    policies: List[str]
    compliance_status: str
    reasoning: str
    tools_used: List[str]
    similar_documents: List[str]  # <- add this


# Create output parser
parser = PydanticOutputParser(pydantic_object=PolicyComplianceResponse)

# Define the system prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a Compliance Auditor Assistant. Your responsibilities include:

            1. Reading the document content.
            2. Extracting relevant entities (e.g., personal data usage, sharing, consent, jurisdictions).
            3. Cross-checking content against policy rules.
            4. Making a compliance decision: **Compliant** or **Non-Compliant**.
            5. Clearly explaining your decision using references from similar documents and the relevant policy.

            ---

            📄 Document Chunk:
            \"\"\"{chunk}\"\"\"

            📚 Policy Rules: Use the `find_matching_policies` tool to retrieve relevant rules.

            📝 Similar Documents: Use the `find_similar_documents` tool to find documents similar to the provided chunk.

            ❓ Compliance Question:
            "Is this contract GDPR compliant?"

            ---

            Wrap your answer in the following structure (no extra text). Use all tools or sources necessary:
            \n{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())


# Create the LLM and agent
llm = get_llm("openai", model_name="gpt-4o")

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[chunk_embedding_tool, matching_policy_tool, similar_document_tool],
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[chunk_embedding_tool, matching_policy_tool, similar_document_tool],
    verbose=True
)

# Gradio execution logic
def process_pdf_and_query(pdf_file, query):
    document_pages = read_pdf(pdf_file.name)
    texts = [page.extract_text() for page in document_pages if page.extract_text()]
    full_text = "\n\n".join(texts)

    response = agent_executor.invoke({"query": query, "chunk": full_text})

    try:
        structured = parser.parse(response.get("output"))
        return f"""🧾 **Compliance Status:** {structured.compliance_status}

                📌 **Policies Used:**
                {chr(10).join(f"- {p}" for p in structured.policies)}

                🛠️ **Tools Used:**
                {chr(10).join(f"- {tool}" for tool in structured.tools_used)}

                📖 **Similar Documents:**
                {chr(10).join(f"- {doc}" for doc in structured.similar_documents)}

                🧠 **Reasoning:**
                {structured.reasoning}
                """
    except Exception as e:
        print("Parser Error:", e)
        print("Raw LLM Output:", response)
        return f"❌ Error occurred while parsing LLM response:\n{e}"


# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 📘 Compliance Auditor")

    with gr.Row():
        pdf_input = gr.File(label="Upload a PDF", file_types=[".pdf", ".PDF"])
        query = gr.Textbox(label="Compliance Question", placeholder="Enter your compliance question here...", lines=2)

    output = gr.Textbox(label="Compliance Report", lines=20, interactive=False)
    read_btn = gr.Button("Run Compliance Audit")

    read_btn.click(fn=process_pdf_and_query, inputs=[pdf_input, query], outputs=output)

app.launch(pwa=True, server_name="0.0.0.0", server_port=8501)
