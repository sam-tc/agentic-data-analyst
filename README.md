# ✦ Lumo Studio

**Your data has a story. Just ask.**

Ever opened a CSV file with hundreds of rows and wondered where to begin? Lumo Studio makes exploring data feel less like writing queries and more like having a conversation.

Upload a dataset, ask a question in plain English, and let Lumo do the analysis—with real Python calculations, interactive charts, and receipts showing exactly what happened behind the scenes.

**[Try Lumo Studio ↗](https://agentic-data-analyst-nbkytjnes7qfrzhhdtdqpb.streamlit.app/)**

---

## What Can Lumo Do?

Lumo isn't just a chatbot that *talks* about data. It can actually work with it.

- **Ask away** — Explore CSV datasets using natural-language questions.
- **Get real answers** — Calculations are performed using Pandas, not guessed by AI.
- **See the bigger picture** — Generate interactive charts with Plotly.
- **Look behind the scenes** — Inspect actual tool calls and their results.
- **Bring your own data** — Upload a CSV or explore the included sample datasets.
- **Keep the analysis going** — When Gemini hits supported API limits or temporary server errors, Groq can take over automatically.

## Six Tools, Plenty of Possibilities

| Tool | What it does |
|---|---|
| Dataset Profile | Understand columns, data types, missing values, and statistics |
| Row Filter | Find records matching specific conditions |
| Aggregation | Calculate sums, averages, medians, counts, and more |
| Group Comparison | Compare results across categories |
| Correlation | Discover linear relationships between numeric columns |
| Visualization | Create bar charts, histograms, scatter plots, and line charts |

All six are actual Python functions. The AI chooses which ones to use, while Python does the calculations.

## See It in Action

Imagine uploading a dataset containing students, their departments, study hours, and marks.

**You ask:**

> Which department has the highest average marks?

**Lumo answers:**

> CSE has the highest average marks at **90**, followed by ECE at 85 and IT at 80.

**Behind the scenes:**

```text
Tool: group_comparison
Group by: department
Operation: mean
Value column: marks

Result:
CSE → 90
ECE → 85
IT  → 80
```

That's the idea behind Lumo: **an answer you can verify**, not just one that sounds convincing.

You can also ask it to visualize order counts by region, calculate correlations, filter records, or summarize unfamiliar datasets.

## Two AI Agents. One Set of Tools.

Lumo uses two language-model providers:

- **Gemini 3.6 Flash** — The primary agent for interpreting questions and choosing tools.
- **Groq GPT-OSS 120B** — The backup agent when Gemini encounters supported rate-limit or temporary service errors.

Both use the same six Python tools, so the underlying calculations remain consistent when they select the same operations.

The flow is simple:

```text
Your question
     ↓
Gemini (primary)
     │
     └── If temporarily unavailable → Groq
                                      ↓
                              Python analytical tools
                                      ↓
                         Answer + charts + receipts
```

The fallback includes a temporary cooldown to avoid repeatedly calling Gemini after certain failures. Both providers still have their own API limits.

## Built With

| Technology | Role |
|---|---|
| Python | Application logic |
| Pandas | Data analysis |
| Plotly | Interactive visualizations |
| Streamlit | Web application |
| Google Gemini | Primary AI agent |
| Groq + GPT-OSS 120B | Backup AI agent |
| Pytest | Automated testing |

## Run Lumo Locally

**1. Clone the repository**

```bash
git clone https://github.com/sam-tc/agentic-data-analyst.git
cd agentic-data-analyst
```

**2. Set up Python**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**3. Configure your API keys**

Copy `.env.example` to `.env`, then add your keys from [Google AI Studio](https://aistudio.google.com/apikey) and [Groq Cloud](https://console.groq.com/).

```dotenv
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-3.6-flash

GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b

LUMO_PROVIDER=gemini
LUMO_GROQ_FALLBACK=true
LUMO_GEMINI_FALLBACK=false
```

Never commit your `.env` file.

**4. Launch Lumo**

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501`, choose a sample CSV or upload your own, and start exploring.

## Tested, Not Just Prompted

Lumo has **90 passing automated tests**, covering analytical calculations, edge cases, function calling, charts, and AI provider switching.

```bash
python -m pytest -q
```

The project also includes two sample datasets—`students.csv` and `sales.csv`—to demonstrate analysis across different data structures.

## A Few Things to Know

Lumo is still a learning project, so there are limitations:

- CSV is the supported upload format, with a 10 MB interface limit.
- AI responses depend on model availability and API quotas.
- The two agents may interpret questions differently, even though they share analytical tools.
- Automatic fallback handles selected failures, not every possible error.
- Chat history is session-based; persistent memory is not implemented.

## What I Learned

Building Lumo helped me understand how an AI agent can go beyond generating text.

The most interesting part was separating the model's decisions from the actual analysis. The AI chooses the tools, but the results come from Python functions that can be tested independently.

I also learned how important error handling, reproducible tests, and API reliability become when moving a project from a local script to a deployed application.

AI assistance was used throughout development for learning, implementation support, testing, and debugging.

---

*Less spreadsheet scrolling. More understanding.*
