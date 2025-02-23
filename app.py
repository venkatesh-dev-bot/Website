from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers.pydantic import PydanticOutputParser
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Define request model
class Question(BaseModel):
    query: str
    model: str = "GPT"  # Default to GPT

# Define response model (for structured parsing)
class AnswerResponse(BaseModel):
    model: str
    answer: str
    source_documents: list[str]

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Specify allowed methods
    allow_headers=["*"],  # Specify allowed headers
)

# Load and process PDF
loader = PyPDFLoader("Venkatesh.pdf")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Store conversation history
chat_history = []

concise_prompt_template = PromptTemplate(
    template="""Given the context below, answer the question concisely and clearly in JSON format:

                Context:
                {context}

                Question:
                {question}

                Answer (in JSON format with 'model', 'answer', and 'source_documents' fields):""",
    input_variables=["context", "question"]
)


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(question: Question):
    try:
        # Select model based on user choice
        if question.model == "GPT":
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)
        elif question.model == "GEMINI":
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                max_retries=2,
                verbose=True
            )
        elif question.model == "MISTRAL":
            llm = ChatMistralAI(
                model="codestral-latest",
                temperature=0,
                max_retries=2,
                verbose=True
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model selection. Choose 'GPT', 'GEMINI', or 'MISTRAL'."
            )

        # Create prompt using the provided template
        prompt = concise_prompt_template.format(
            context=" ".join([doc.page_content for doc in documents]),
            question=question.query
        )

        # Define the Pydantic parser
        pydantic_parser = PydanticOutputParser(pydantic_object=AnswerResponse)
        
        # Get the format instructions for parser
        format_instructions = pydantic_parser.get_format_instructions()

        # Create the LLM chain using the chain style
        chain = concise_prompt_template | llm | pydantic_parser 

        # Execute the chain asynchronously
        result = await chain.ainvoke({
            "context": " ".join([doc.page_content for doc in documents]),
            "question": question.query
        })

        # Update chat history
        chat_history.append((question.query, result.answer))

        # Return response in a structured format
        return JSONResponse(content={
            "model": question.model,
            "answer": result.answer,
            "source_documents": result.source_documents
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Resume QA Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
