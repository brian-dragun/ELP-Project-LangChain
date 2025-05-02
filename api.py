from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from typing import Optional, List
import os

# Import your existing components
from reasoning import ReasoningService
from ai_agent import llm
import ingest_documents

# Initialize FastAPI application
app = FastAPI(
    title="ELP Document Intelligence API",
    description="API for querying and analyzing documents across multiple office locations",
    version="1.0.0"
)

# Define request/response models
class QueryRequest(BaseModel):
    query: str
    reasoning_mode: bool = False
    detail_level: str = "standard"

class RefreshRequest(BaseModel):
    documents_path: str = "./documents"
    full_rebuild: bool = False

class PatternRequest(BaseModel):
    text: str
    pattern_type: str  # currency, percentage, date, email, phone, time, numeric

class SummaryRequest(BaseModel):
    document_path: str
    max_length: Optional[int] = 500

class VisualizationRequest(BaseModel):
    data: str
    chart_type: Optional[str] = None
    title: Optional[str] = None

class CompareRequest(BaseModel):
    documents: List[str]
    aspects: Optional[List[str]] = None

class FactRequest(BaseModel):
    statement: str
    source: str = "API"
    confidence: float = 0.8

class InsightRequest(BaseModel):
    query: Optional[str] = None
    documents: Optional[List[str]] = None
    insight_type: Optional[str] = "general"  # general, cost, utilization, sustainability

# Initialize reasoning service
reasoning_service = ReasoningService(llm=llm, chroma_path="./chroma_db")

# Basic query endpoint
@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a document query and return results with reasoning"""
    try:
        result = reasoning_service.process_query(
            request.query, 
            detail_level="detailed" if request.reasoning_mode else "standard"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Document management endpoints
@app.post("/documents/refresh")
async def refresh_documents(request: RefreshRequest, background_tasks: BackgroundTasks):
    """Refresh the document database with any changes"""
    try:
        if request.full_rebuild:
            background_tasks.add_task(
                ingest_documents.ingest_documents,
                docs_dir=request.documents_path,
                db_path="./chroma_db"
            )
            return {"status": "Full document database rebuild started"}
        else:
            background_tasks.add_task(
                ingest_documents.update_database_incrementally,
                documents_dir=request.documents_path,
                chroma_path="./chroma_db"
            )
            return {"status": "Incremental document update started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/status")
async def document_status():
    """Get document database status and stats"""
    try:
        # Basic implementation to get stats
        doc_count = sum(1 for f in os.listdir("./documents") 
                      if os.path.isfile(os.path.join("./documents", f)))
        
        return {
            "document_count": doc_count,
            "database_path": "./chroma_db",
            "last_updated": os.path.getmtime("./chroma_db") if os.path.exists("./chroma_db") else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Fact management endpoints
@app.get("/facts")
async def get_facts():
    """Get all extracted facts"""
    try:
        # Access fact memory
        from ai_agent_interactive_chat import FactMemory
        fact_memory = FactMemory()
        facts = fact_memory.facts + fact_memory.session_facts
        return {"facts": facts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/facts")
async def add_fact(request: FactRequest):
    """Add a new fact to the fact memory"""
    try:
        from ai_agent_interactive_chat import FactMemory
        fact_memory = FactMemory()
        fact_memory.add_fact({
            "fact": request.statement,
            "source": request.source,
            "confidence": request.confidence
        })
        return {"status": "success", "message": "Fact added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pattern extraction endpoint
@app.post("/extract-patterns")
async def extract_patterns(request: PatternRequest):
    """Extract patterns from text (dates, emails, numbers, etc)"""
    try:
        from ai_agent_interactive_chat import PatternExtractor
        extractor = PatternExtractor()
        patterns = extractor.extract_pattern(request.text, request.pattern_type)
        return {"patterns": patterns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Document summarization endpoint
@app.post("/summarize")
async def summarize_document(request: SummaryRequest):
    """Generate a summary of a document"""
    try:
        from ai_agent_interactive_chat import DocumentSummarizer
        summarizer = DocumentSummarizer()
        summary = summarizer.summarize_document(request.document_path, max_length=request.max_length)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Visualization generation endpoint
@app.post("/visualize")
async def create_visualization(request: VisualizationRequest):
    """Generate a visualization from data"""
    try:
        from ai_agent_interactive_chat import DataVisualizer
        visualizer = DataVisualizer(output_dir="./visualizations")
        result = visualizer.create_visualization(
            request.data, 
            chart_type=request.chart_type, 
            title=request.title
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Cross-document analysis endpoint
@app.post("/compare-documents")
async def compare_documents(request: CompareRequest):
    """Compare multiple documents and find similarities/differences"""
    try:
        from ai_agent_interactive_chat import CrossDocumentAnalyzer
        analyzer = CrossDocumentAnalyzer()
        comparison = analyzer.compare_documents(request.documents, aspects=request.aspects)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Insight generation endpoint
@app.post("/insights")
async def generate_insights(request: InsightRequest):
    """Generate insights from documents based on optional query"""
    try:
        from ai_agent_interactive_chat import InsightGenerator
        generator = InsightGenerator()
        insights = generator.generate_insights(
            query=request.query, 
            documents=request.documents,
            insight_type=request.insight_type
        )
        return {"insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

# API documentation redirect
@app.get("/")
async def root():
    """Redirect to API documentation"""
    return {"message": "Welcome to ELP Document Intelligence API. Visit /docs for interactive documentation."}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)