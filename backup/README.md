# Document Q&A System with LangChain, LangGraph and Lambda Labs

This system allows you to ingest documents and ask questions about them using LangChain, LangGraph, and Lambda Labs API.

## Setup

1. Install the required packages:
   python3 -m venv .venv && source .venv/bin/activate
   ```
   pip install -r requirements.txt
   ```

2. Make sure your documents are placed in the `documents/` folder.

3. Run the document ingestion process:
   ```
   python ingest_documents.py
   ```

## Usage

You can ask questions about your documents in two ways:

### Using the ask_question.py script

```bash
./ask_question.py "Your question about the documents?"
```

For example:
```bash
./ask_question.py "What was the average meeting room utilization in NYC?"
```

### Using the qa_system.py directly

Edit the question in `qa_system.py` and run:
```bash
python qa_system.py
```

## System Components

- `ingest_documents.py`: Processes documents and creates a vector database
- `qa_system.py`: Main system that handles retrieval and question answering
- `ask_question.py`: Command-line interface for asking questions



# Sample Questions for Office Data Analysis

## Sample Questions

- "How many employees work in the Philadelphia office?"
- "What is the lease term for the Los Angeles office?"
- "What are the ESG metrics for the Miami office?"
- "Compare meeting room utilization rates across all cities."
- "Which office has the highest energy consumption?"

## Current Data Analysis Questions

### Badge & Employee Data
- "How many employees work in the Los Angeles office?"
- "What is the badge utilization rate in Chicago compared to NYC?"
- "Which city office has the highest employee count?"
- "How does employee turnover rate compare between Philadelphia and Miami?"
- "What is the average workforce diversity percentage across all offices?"

### Office Space & Facilities
- "What is the meeting room utilization rate in Los Angeles on weekdays?"
- "Which city has the highest meeting room utilization on weekends?"
- "What is the vacancy rate for the NYC office?"
- "How does office space cost per square foot in Philadelphia compare to the national average?"
- "What is the total square footage of our lease in Los Angeles?"

### ESG & Sustainability Metrics
- "Which office has the highest energy consumption per seat?"
- "What is the LEED certification level for the Chicago office?"
- "How does water usage compare between the Miami and NYC offices?"
- "What is the waste recycling rate in Philadelphia?"
- "Which office has the best overall ESG metrics?"

### Financial & Lease Data
- "When does the Los Angeles office lease end?"
- "What is the build-out cost per square foot in NYC?"
- "Which city has the highest rent per square foot?"
- "What is the executive compensation ratio across all offices?"
- "What is the total occupancy rate across all office locations?"

## Future Prediction Questions

### Short-term Predictions
- "Based on current trends, what will the meeting room utilization in NYC be by the end of 2025?"
- "Given the current vacancy rates, how might the Philadelphia office occupancy change in the next 6 months?"
- "What is the projected energy consumption for the Chicago office in Q3 2025?"
- "How might the LA office badge utilization rate change by December 2025?"
- "Based on current data, will the Miami office need additional meeting rooms by the end of 2025?"

### Long-term Predictions
- "Given current growth rates, will we need to expand our square footage in NYC by 2027?"
- "Based on ESG improvement trends, when might the Miami office achieve Gold LEED certification?"
- "What is the projected employee turnover rate in Chicago for 2026?"
- "How will rent costs in Philadelphia likely change when the current lease approaches renewal?"
- "Based on current workforce diversity trends, what will our company-wide diversity percentage be by 2028?"

### Market & Comparative Analysis
- "How does our office space cost in LA compare to predicted market rates for 2026?"
- "Based on current market trends, which of our city locations will see the highest rent increases by 2027?"
- "Given current utilization patterns, which office location would benefit most from a hybrid work policy by 2026?"
- "How will our overall real estate portfolio performance compare to industry benchmarks in 2027?"
- "Which of our office locations is projected to have the best ROI over the next 5 years?"

## Complex Multi-factor Questions

- "Considering both ESG metrics and office costs, which location offers the best balance of sustainability and financial efficiency?"
- "Based on badge utilization, meeting room usage, and employee count, which office is most efficiently using its space?"
- "How does employee satisfaction correlate with office sustainability metrics across our locations?"
- "Comparing lease terms, occupancy rates and forecasted growth, which office might need renegotiation first?"
- "Based on all available metrics, which office location should be our model for future expansions?"