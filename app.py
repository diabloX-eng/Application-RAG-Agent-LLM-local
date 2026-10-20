import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import langchainhub as hub
# --- Imports sécurisés des outils et agents ---
try:
    from langchain.tools.retriever import create_retriever_tool
except ImportError:
    from langchain_core.tools import create_retriever_tool

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent

st.set_page_config(page_title="Assistant RAG & Agentique PDF", layout="wide")
st.title("📄 Assistant RAG & Agentique sur PDF")

# Initialisation de la session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

# Template du prompt ReAct défini localement
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# Barre latérale pour charger le PDF
with st.sidebar:
    st.header("📂 Importer un document")
    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])

    if uploaded_file is not None and st.button("Traiter le PDF"):
        with st.spinner("Analyse et indexation du PDF en cours..."):
            # 1. Sauvegarde temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # 2. Chargement du PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # 3. Découpage en blocs (Chunks)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(docs)

            # 4. Embeddings et Vector Store
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

            # 5. Création du Retriever et de l'outil Agent
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            tool = create_retriever_tool(
                retriever,
                name="recherche_pdf",
                description="Recherche des informations précises à l'intérieur du document PDF."
            )
            tools = [tool]

            # 6. Chargement du LLM et de l'Agent
            llm = Ollama(model="llama3")
            agent = create_react_agent(llm, tools, prompt)
            st.session_state.agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True # Permet de gérer les erreurs de parsing du LLM
)
            os.remove(tmp_path)
            st.success("✅ PDF indexé avec succès ! Vous pouvez poser vos questions.")

# Affichage de l'historique du chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
query = st.chat_input("Posez votre question sur le PDF...")

if query:
    if st.session_state.agent_executor is None:
        st.warning("⚠️ Veuillez téléverser et traiter un fichier PDF dans la barre latérale.")
    else:
        st.chat_message("user").markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("L'agent cherche la réponse..."):
                response = st.session_state.agent_executor.invoke({"input": query})["output"]
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})