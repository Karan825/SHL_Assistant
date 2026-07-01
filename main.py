from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from agent.retriever import Retriever
from agent.agent import SHLAgent
from dotenv import load_dotenv

load_dotenv()

# --- Pydantic Models ---
class Message(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    messages: list[Message]
class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str
class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

# --- App Startup ---
retriever = None
agent = None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global retriever, agent
    retriever = Retriever()
    agent = SHLAgent(retriever)
    yield
    retriever = None
    agent = None

app = FastAPI(lifespan=lifespan)

# --- Endpoints ---
@app.get("/health")
def health():
    return {'status':'ok'}

@app.post('/chat', response_model= ChatResponse)
def chat(request: ChatRequest):
    messages = [{'role':m.role,'content':m.content} for m in request.messages]
    if not messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    result = agent.chat(messages)
    return ChatResponse(**result)