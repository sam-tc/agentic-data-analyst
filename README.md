# Lumo Studio

### An AI-powered, tool-using data analysis workspace

Lumo Studio is a Python application that allows users to upload CSV datasets, ask questions in natural language, and receive answers supported by real analytical calculations.

The system uses Google's Gemini API for understanding questions and selecting tools. The actual data analysis is performed by six Python tools built with Pandas and Plotly.

Unlike a basic LLM chatbot, Lumo maintains an execution trace showing which tools were called, what arguments they received, and what results they produced.

## Live Application

Try Lumo Studio:
https://agentic-data-analyst-nbkytjnes7qfrzhhdtdqpb.streamlit.app/

## Features

- Upload and explore CSV datasets.
- Ask analytical questions in natural language.
- Automatically select and execute analytical tools.
- Generate interactive data visualizations.
- Inspect tool execution history.
- View dataset statistics, column types, missing values, and available correlation heatmaps.
- Use dataset-aware suggested questions and quick tool actions.
- Analyze different datasets without hard-coded column names.
- Handle invalid inputs and temporary API errors.
- Run automated tests without consuming API quota.

## The Six Analytical Tools

| Tool | Purpose |
|---|---|
| Dataset Profiler | Inspects columns, data types, missing values, unique values, and numeric statistics. |
| Row Filtering | Selects rows using comparisons such as greater than, equal to, and text matching. |
| Statistical Aggregation | Calculates count, sum, mean, median, minimum, and maximum for a column. |
| Grouping and Comparison | Groups records by category and compares statistics between groups. |
| Correlation Analysis | Calculates Pearson correlation between two numerical columns. |
| Data Visualization | Generates interactive bar charts, histograms, scatter plots, and line charts. |

These tools perform distinct operations and can be combined by the AI agent when a question requires multiple analytical steps.

## How the Agent Works

1. The user uploads a CSV dataset.
2. Pandas loads the dataset into a DataFrame.
3. The user submits a natural-language question.
4. Gemini examines the available columns and registered tool descriptions.
5. The model selects a suitable tool and supplies its arguments.
6. The Python tool executes the actual calculation.
7. The tool result is returned to the model.
8. The model generates an answer using the calculated result.
9. Lumo displays the answer, any generated charts, and the execution trace.

For example:

**Question:** Which department has the highest average marks?

**Selected tool:** `group_comparison`

**Arguments:**

```json
{
    "group_column": "department",
    "operation": "mean",
    "value_column": "marks"
}
```

**Result:** CSE has the highest average, with 90 marks, in the included student example.

## Technology Stack

- Python
- Pandas — dataset processing and statistical calculations
- Plotly — interactive visualizations
- Streamlit — web interface
- Google Gemini API — natural-language understanding and function calling
- Google GenAI Python SDK — model integration
- Pytest — automated testing
- Git and GitHub — version control

## Project Structure

```text
agentic-data-analyst/
├── app.py
├── run_agent.py
├── requirements.txt
├── README.md
├── .env.example
├── .streamlit/
│   └── config.toml
├── assets/
│   └── styles.css
├── sample_data/
│   ├── students.csv
│   └── sales.csv
├── src/
│   ├── agent.py
│   ├── agent_tools.py
│   ├── data_loader.py
│   └── tools/
│       ├── profile.py
│       ├── filtering.py
│       ├── aggregation.py
│       ├── grouping.py
│       ├── correlation.py
│       └── visualization.py
└── tests/
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sam-tc/agentic-data-analyst.git
cd agentic-data-analyst
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it using:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API

Create an API key using Google AI Studio:

https://aistudio.google.com/apikey

Copy the example environment file:

```bash
cp .env.example .env
```

Add your key to `.env`:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

Use a model available to your Google AI Studio project. Free-tier eligibility, quotas, and availability depend on Google's current policies.

Never commit your `.env` file or API keys.

## Running the Web Application

Start the Streamlit server:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

http://localhost:8501

The application selects `students.csv` by default. You can switch to another included sample dataset or upload your own CSV through the sidebar.

The Streamlit interface currently limits uploads to 10 MB.

## Running from the Terminal

You can also ask questions without the web interface:

```bash
python run_agent.py sample_data/students.csv "Which department has the highest average marks?"
```

For another dataset:

```bash
python run_agent.py sample_data/sales.csv "Which region generated the highest total revenue?"
```

The terminal displays the generated answer and the actual tool calls.

## Testing

Run the entire automated test suite:

```bash
python -m pytest -v
```

Tests cover:

- Loading valid and empty CSV datasets
- Dataset profiling
- Numeric and text filtering
- Statistical aggregation
- Group comparisons
- Pearson correlation
- Chart generation
- Invalid input handling
- Tool execution traces
- Agent behavior using mocked Gemini responses
- Rejection of analytical answers without tool execution

Mocked agent tests do not require live Gemini API requests.

## Demonstration

Two example datasets are included.

**Student dataset**

- Compare average marks across departments.
- Calculate correlation between study hours and marks.
- Create a chart of student counts by department.

**Sales dataset**

- Compare total revenue across regions.
- Find the highest-revenue region.
- Visualize order counts by region.

These examples demonstrate that the system is not restricted to one dataset structure.

## Design Decisions

**Why separate the LLM from the analytical tools?**

The LLM is responsible for interpreting questions and selecting tools. Pandas performs the numerical calculations. This makes the analysis easier to test and inspect.

**Why maintain execution traces?**

A plausible AI-generated answer does not prove that analysis occurred. The trace records real Python tool executions and their arguments and results.

**Why use Streamlit?**

Streamlit provides a Python-based interactive interface without requiring a separate frontend and backend server.

**Why use Plotly?**

Plotly supports interactive charts that users can inspect directly in the browser.

## Limitations

- The current interface supports CSV uploads.
- It requires access to an eligible Gemini API project.
- API availability and rate limits may affect responses.
- Statistical operations require suitable column types.
- Very large datasets may require additional performance improvements.
- Chat history is retained within the current Streamlit session; persistent cross-session agent memory has not been implemented.

## Development and AI Assistance

AI assistance was used during planning, explanation, implementation, and debugging.

The project was built incrementally, with Git checkpoints and automated tests for individual tools and agent behavior. The implementation uses custom Python analytical functions rather than relying on the LLM to generate numerical answers without executing tools.

## Future Improvements

- Persistent memory for user-defined metrics and preferences
- More advanced multi-step analytical workflows
- Additional file formats
- Exportable analysis reports
- More advanced visualization options
