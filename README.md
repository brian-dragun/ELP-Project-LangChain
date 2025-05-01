# Document Q&A System with LangChain, LangGraph and Lambda Labs

This system allows you to ingest documents and ask questions about them using LangChain, LangGraph, and Lambda Labs API.

## Enterprise Office Utilization and ESG Simulation

This project includes a comprehensive simulation of corporate office behaviors across multiple U.S. cities for an insurance company. The simulation focuses on occupancy trends, energy consumption, ESG performance, and strategic space optimization.

### Cities Included:
- Philadelphia
- Los Angeles
- Miami
- Chicago
- New York City

### Data Categories Simulated:
- Employee Directory
- Badge Swipe Data
- Meeting Room Utilization
- Lease and Market Data
- Energy Consumption
- ESG Metrics

## Setup

1. Install the required packages:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Make sure your documents are placed in the `documents/` folder.

3. Run the document ingestion process:
   ```bash
   python ingest_documents.py
   ```

4. Set up environment variables for API keys and configuration:
   - Create a `.env` file in the project root
   - Add your API keys and configuration settings
   - See the `.env.example` file for required variables

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

### Interactive Chat (v3)

For a more interactive experience, use the v3 interactive chat:
```bash
python interactive_chat_v3.py
```

## System Components

- `ingest_documents.py`: Processes documents and creates a vector database
- `qa_system.py`: Main system that handles retrieval and question answering
- `ask_question.py`: Command-line interface for asking questions
- `config.py`: Configuration management for API keys and settings
- `interactive_chat_v3.py`: Interactive chat interface for asking questions

## Simulation Details

### Key Simulation Assumptions:
- Badge Utilization Rates: City-specific % of employees badging in daily
- Door Usage: Main door gets 50% of swipes; other doors distributed randomly
- Entry vs Exit: 70% Entry; 30% Exit
- Energy Consumption: Based on energy per seat per year scaled by badge activity
- Occupancy Thresholds: Over 90% occupancy triggers "Over Threshold" warning
- Electricity Costs: Average $/kWh set per city
- Meeting Room Booking: Higher utilization on weekdays
- Environmental Impact: Calculated from occupancy-driven energy use
- Employee Turnover: Incorporated into Social ESG metrics

### Simulated Advanced Analysis:
- Occupancy Threshold Monitoring
- Daily Energy Cost Estimates
- Future Occupancy Predictions
- Hybrid Work Cost Savings Simulation
- Office Consolidation Risk Identification
- Strategic Resource Allocation Modeling

### Files Generated:
- *_employees.csv: Employee directories for each city
- *_badge_data.csv: Badge swipes with multiple doors
- *_meeting_room_utilization.csv: Meeting room usage data
- *_lease_market_data.csv: Lease and vacancy data
- *_esg_metrics.csv: ESG metrics per city
- occupancy_energy_analysis_v2.csv: Occupancy % and daily energy costs

## AI Agent Capabilities

Your AI Agent can now support:
- Real-time Reporting
- Occupancy Forecasting
- Energy Forecasting
- Strategic Space Planning
- ESG Recommendations

## Sample Questions

### Occupancy & Space Utilization
- "Which office had the highest average occupancy in April 2025?"
- "How many days did Philadelphia exceed safe occupancy thresholds?"
- "Predict which offices will face over-capacity issues by end of next quarter."

### Energy Cost Analysis
- "Which city has the highest average daily energy cost?"
- "How much are we spending annually on energy in the Los Angeles office?"
- "Predict the total energy cost for the NYC office for the next 6 months."

### ESG and Environmental Impact
- "Which office has the most environmentally friendly performance based on energy usage?"
- "Where could implementing stricter occupancy policies reduce energy costs the most?"

### Strategic Office Planning
- "Which city would benefit most from downsizing office space based on under-utilization?"
- "If we relocated 25% of New York staff to Miami, what would be the estimated occupancy impact?"

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

## Ready-to-Use Prompts for Your AI Agent

### Predict Future Headcount Needs
```
"Based on the current occupancy rates across all offices, predict which offices will need to grow their headcount in the next 6 months. Assume that offices operating above 80% average occupancy will require a 10% increase in seats to maintain safe thresholds. List the offices and estimated new headcounts."
```

### Recommend Office Consolidation
```
"Analyze occupancy data and recommend if any offices should be consolidated due to underutilization. Flag any city where average occupancy is below 50%. Provide a consolidation recommendation and estimated annual savings if offices are closed."
```

### Predict Occupancy for Next Month
```
"Using historical badge swipe and occupancy trends from April 2025, predict the expected average occupancy for each office in May 2025. Highlight any offices expected to exceed 85% average occupancy."
```

### Recommend Cost Savings Strategies
```
"Review occupancy, lease costs, and energy costs for all offices. Recommend at least three cost savings strategies. Prioritize actions like moving underutilized offices to smaller spaces, shifting to hybrid models, or targeting energy efficiency improvements."
```

## Complex Multi-factor Questions

- "Considering both ESG metrics and office costs, which location offers the best balance of sustainability and financial efficiency?"
- "Based on badge utilization, meeting room usage, and employee count, which office is most efficiently using its space?"
- "How does employee satisfaction correlate with office sustainability metrics across our locations?"
- "Comparing lease terms, occupancy rates and forecasted growth, which office might need renegotiation first?"
- "Based on all available metrics, which office location should be our model for future expansions?"