# Schema-Aware Text-to-SQL System

A multi-step Text-to-SQL pipeline using LangChain and LangGraph with support for multiple LLMs and Spider benchmark evaluation.

## Features

✅ **Multi-step Chain**: Question → SQL Generation → Execution → Natural Language Answer  
✅ **Schema-Aware**: Validates table names and includes schema context in prompts  
✅ **Multiple LLMs**: Compare Groq, Ollama, Cohere, Gemini models  
✅ **Error Handling**: SQL validation with QuerySQLCheckerTool  
✅ **Feedback Loop**: User thumbs up/down with query regeneration  
✅ **Spider Benchmark**: Exact Match and Execution Accuracy evaluation  
✅ **CLI Interface**: Interactive and batch modes  

## Installation

```bash
pip install -r req.txt
```

## Spider Dataset Setup

### Structura necesară:
```
spider/
├── dev.json              # Întrebări de dezvoltare
├── train.json            # Întrebări de antrenare
├── tables.json           # Metadate tabele
├── database/             # Baze de date SQLite (descarcă separat!)
│   ├── concert_singer/
│   ├── car_1/
│   └── ...
└── evaluation.py         # Script oficial evaluare (opțional)
```

### Descarcă bazele de date:

**IMPORTANT**: Repository-ul Spider de pe GitHub **NU** include bazele de date!

```bash
cd spider
wget https://drive.google.com/uc?export=download&id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m -O database.zip
unzip database.zip
```

Sau manual:
1. Descarcă: https://drive.google.com/uc?export=download&id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m
2. Extrage în `spider/database/`

### Testează configurația:
```bash
python test_spider.py
```

## Spider2-Lite Setup

Spider2-Lite este versiunea modernă, cu suport pentru SQLite local.

### Descarcă bazele de date:

```bash
cd spider2-lite/resource/databases
mkdir -p spider2-localdb
cd spider2-localdb

# Descarcă și extrage
wget https://drive.usercontent.google.com/download?id=1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG -O databases.zip
unzip databases.zip
```

### Testează:
```bash
python test_spider2_lite.py
```

### Evaluare Spider2-Lite:
```bash
# Primele 5 întrebări
python main.py --spider2-lite ./spider2-lite --max-questions 5

# Toate întrebările SQLite
python main.py --spider2-lite ./spider2-lite
```

## Configuration

Create `.env` file:
```
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```

## Usage

### Basic Query
```bash
python main.py --question "How many movies are available?"
```

### Interactive Mode
```bash
python main.py --interactive
```

### Compare LLMs
```bash
python main.py --compare --question "List all directors"
```

### With Feedback Loop
```bash
python main.py --interactive --feedback
```

### Spider Benchmark Evaluation
```bash
# Evaluate on first 10 questions
python main.py --spider-benchmark ./spider --max-questions 10

# Evaluate on all dev set (takes longer)
python main.py --spider-benchmark ./spider
```

## Architecture

```
User Question
     ↓
[write_query] → Generate SQL using LLM + Schema Context
     ↓
[validate_tables] → Check table names exist
     ↓
[execute_query] → Run SQL with validation
     ↓
[generate_answer] → Convert results to natural language
     ↓
Final Answer
```

## Spider Evaluation Metrics

- **Exact Match Accuracy**: Normalized SQL string comparison
- **Execution Accuracy**: Result set comparison (tests if queries produce same results)
- **By Difficulty**: Performance breakdown by easy/medium/hard/extra difficulty levels

### Example Output
```
📊 SPIDER BENCHMARK RESULTS
============================================================
Exact Match Accuracy: 45.50%
Execution Accuracy: 68.20%
Total Questions: 100

📈 By Difficulty Level:

EASY:
  Count: 30
  Exact Match: 63.33%
  Execution: 80.00%

MEDIUM:
  Count: 40
  Exact Match: 42.50%
  Execution: 65.00%

HARD:
  Count: 20
  Exact Match: 25.00%
  Execution: 50.00%
```

## Dependencies

- LangChain + LangGraph for orchestration
- Groq/Ollama/Cohere/Gemini for LLM inference
- SQLAlchemy for database operations
- Spider dataset for evaluation

## Project Structure

```
schema-aware-text2sql/
├── main.py                 # Main pipeline
├── spider_evaluator.py     # Spider benchmark evaluator
├── test_spider.py          # Test Spider setup
├── db/                     # Your custom databases
│   └── netflixdb.sqlite
├── spider/                 # Spider benchmark dataset
│   ├── dev.json
│   ├── train.json
│   ├── tables.json
│   └── database/
├── req.txt                 # Dependencies
├── .env                    # API keys
└── README.md
```

## Next Steps

1. **Test the Spider Dataset Integration**:
   - Run the following command to evaluate your pipeline on the Spider dataset:
     ```bash
     python main.py --spider-benchmark ./spider --max-questions 10
     ```
   - This will evaluate your pipeline on the first 10 questions in the `dev.json` file.

2. **Analyze Results**:
   - The evaluation results will include metrics like Exact Match Accuracy and Execution Accuracy.
   - Use these metrics to identify areas for improvement in your pipeline.

3. **Optimize Your Pipeline**:
   - Use the feedback loop and error-handling mechanisms to improve the accuracy of your pipeline.
   - Experiment with different LLMs and prompt engineering techniques to improve SQL generation.

4. **Document Your Work**:
   - Update your `README.md` with instructions on how to use the Spider dataset for evaluation.
   - Include the evaluation results and any insights you gained from the analysis.

## Spider Dataset Evaluation

To evaluate the Text-to-SQL pipeline on the Spider dataset:

1. Download the Spider dataset from the official repository: [Spider Dataset](https://yale-lily.github.io/spider).
2. Place the dataset in the `spider/` directory:
   ```
   schema-aware-text2sql/
   ├── spider/
   │   ├── dev.json
   │   ├── train.json
   │   ├── tables.json
   │   └── database/
   │       ├── concert_singer/
   │       ├── car_1/
   │       └── ...
   ```
3. Run the evaluation:
   ```bash
   python main.py --spider-benchmark ./spider --max-questions 10
   ```

4. View the results:
   ```
   📊 SPIDER BENCHMARK RESULTS
   ============================================================
   Exact Match Accuracy: 45.50%
   Execution Accuracy: 68.20%
   Total Questions: 10
   ```

5. The results will also be saved to `spider_evaluation_results.json` in the project directory.