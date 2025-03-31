from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from fastapi import FastAPI
from PyPDF2 import PdfReader
import os
import requests
import re

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = "/home/ubuntu/ApexDeveloperGuidea.pdf"
print("OPENAI_API_KEY :{OPENAI_API_KEY}")
# Salesforce credentials
SALESFORCE_INSTANCE_URL = os.getenv("SALESFORCE_INSTANCE_URL") 
SALESFORCE_ACCESS_TOKEN = os.getenv("SALESFORCE_ACCESS_TOKEN")
SALESFORCE_API_VERSION = "v60.0"
LEAD_ENDPOINT = f"{SALESFORCE_INSTANCE_URL}/services/data/{SALESFORCE_API_VERSION}/sobjects/Lead/"
print("End Point:",LEAD_ENDPOINT)
if not OPENAI_API_KEY:
    raise ValueError("ERROR: OPENAI_API_KEY is missing from .env file!")

if not SALESFORCE_ACCESS_TOKEN:
    raise ValueError("ERROR: SALESFORCE_ACCESS_TOKEN is missing from .env file!")

# Load PDF and create vector store
def get_vectorstore_from_static_pdf(pdf_path=PDF_PATH):
    pdf_reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store

# Load vector store at startup
vector_store = get_vectorstore_from_static_pdf()

# Chat request model
class ChatRequest(BaseModel):
    message: str

# Store chat history
chat_history = []

# Function to extract lead details from user input
def extract_lead_details(user_input):
    name_match = re.search(r"name is (\w+ \w+|\w+)", user_input, re.IGNORECASE)
    # Mobile number, phone number, and email regex
    mobile_match = re.search(r"mobile number is ([\d\s\-]+)", user_input, re.IGNORECASE)
    phone_match = re.search(r"phone number is ([\d\s\-]+)", user_input, re.IGNORECASE)
    email_match = re.search(r"email is ([\w\.-]+@[\w\.-]+\.\w+)", user_input, re.IGNORECASE)

    first_name, last_name = None, None
    if name_match:
        name_parts = name_match.group(1).split()
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""

    # Ensure at least one contact number is provided
    mobile_number = mobile_match.group(1) if mobile_match else None
    phone_number = phone_match.group(1) if phone_match else None
    email = email_match.group(1) if email_match else None

    if not last_name:
        last_name = first_name

    # Assign phone number to mobile if mobile is None
    if not mobile_number:
        mobile_number = phone_number

    # Require at least a mobile number
    if not mobile_number:
        return "Please provide a mobile number to create the lead."

    if last_name:
        return {
            "FirstName": first_name,
            "LastName": last_name,
            "Company": "FSTC",
            "Status": "Open - Not Contacted",
            "Email": email,
            "MobilePhone": mobile_number  # Only using mobile number as per requirement
        }
    return None


# Function to create Lead in Salesforce
def create_salesforce_lead(lead_data):
    headers = {
        "Authorization": f"Bearer {SALESFORCE_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(LEAD_ENDPOINT, json=lead_data, headers=headers)
    print(f"Salesforce API Response: {response.status_code}, {response.text}")  # 🔍 Debugging
    if response.status_code == 201:
        return f"Lead '{lead_data['FirstName']} {lead_data['LastName']}' created successfully in Salesforce."
    else:
        print(f"Salesforce API Response: {response.status_code}, {response.text}")
        return f"Invalid input. Please provide Mobile number."

# Function to get response based on chat history
def get_response(user_input):
    global chat_history
    print(f"User Input: {user_input}")

    # Check if user wants to create a lead
    if "lead" in user_input.lower() and ("create" in user_input.lower() or "insert" in user_input.lower()):
        print("Detected Lead Creation Request")
        lead_details = extract_lead_details(user_input)
        if lead_details:
            print(f"Extracted Lead Details: {lead_details}")
            return create_salesforce_lead(lead_details)
        else:
            return "Invalid input. Please provide Mobile number."

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    documents = retriever.get_relevant_documents(user_input)

    if not documents:  
        return "I don't know. The PDF does not contain relevant information."

    # Create conversation-aware response
    context = "\n".join([doc.page_content for doc in documents])
    chat_history.append(HumanMessage(content=user_input))

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    response = llm.invoke(
        f"Answer the question based **only** on the provided context. "
        f"If the answer is not in the context, say 'I don't know'.\n\n"
        f"**Context:**\n{context}\n\n"
        f"**Chat History:**\n{chat_history}\n\n"
        f"**User's Question:** {user_input}"
    )

    chat_history.append(AIMessage(content=response.content))
    return response.content

# API Endpoint
@app.post("/api/govind")
async def chat_endpoint(chat_request: ChatRequest):
    response = get_response(chat_request.message)
    print("response :{response}")
    return {"answer": response}
