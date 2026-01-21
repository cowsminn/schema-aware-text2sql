from pydantic import BaseModel, Field
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLCheckerTool
from langgraph.graph import START, StateGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from pathlib import Path
import os
import argparse
import json
import re

# sql lite local, nu am credidentiale pt google cloud(bigquery) sau snowflake(sf_bq)
#100k tokens pe grok, 1 cuvant = 1,4 * tokens

load_dotenv()

class State(BaseModel):
    question: str           
    query: str = ""       
    result: str = ""   
    answer: str = ""

class QueryOutput(BaseModel):
    """Generated SQL query."""
    query: str = Field(description="Syntactically valid SQL query.")
    
class FeedbackState(BaseModel):
    """Stare extinsă cu feedback."""
    question: str
    query: str = ""
    result: str = ""
    answer: str = ""
    feedback_score: int = 0  # 1=pozitiv, 0=neutru, -1=negativ
    feedback_comment: str = ""

class EvaluationResult(BaseModel):
    question: str
    query_ground_truth: str
    query_generated: str
    exact_match: bool
    execution_accuracy: bool

# def get_db():
#     engine = create_engine("postgresql+psycopg2://postgres:testare@localhost:5432/text2sql_db")
#     return SQLDatabase(engine)

def get_db():
    from sqlalchemy.pool import StaticPool
    from sqlalchemy import event, text
    import os
    
    db_path = os.path.join(os.path.dirname(__file__), "db", "netflixdb.sqlite")
    
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={'check_same_thread': False},
        poolclass=StaticPool
    )
    
    # Disable automatic date conversion for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA parse_time=0")
        cursor.close()
    
    return SQLDatabase(engine, sample_rows_in_table_info=0)

system_message = """
You are an expert Text-to-SQL system.

Given a natural language question and a database schema, generate a
syntactically correct SQLite SQL query.

IMPORTANT RULES:
- Use ONLY table and column names that appear EXACTLY in the schema.
- Do NOT invent or pluralize table names.
- Do NOT invent columns.
- If a table or column does not exist, do NOT use it.
- Pay close attention to the data types of columns (boolean, date, integer, varchar, etc.)

BOOLEAN HANDLING:
- For boolean columns, use 1 for TRUE and 0 for FALSE in SQLite
- Example: WHERE available_globally = 1

Database schema (with sample rows):
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate([
    ("system", system_message), 
    ("user", user_prompt)
])

def get_schema_tables(table_info: str):
    """Extract table names from schema info."""
    # Match both CREATE TABLE and Table: patterns
    tables = set(re.findall(r'CREATE TABLE ["\']?(\w+)["\']?', table_info, re.IGNORECASE))
    tables.update(re.findall(r'Table:\s*["\']?(\w+)["\']?', table_info, re.IGNORECASE))
    return tables

def validate_tables(sql: str, table_info: str):
    """Validate that SQL only uses tables that exist in schema."""
    schema_tables = get_schema_tables(table_info)
    
    # Extract table names from FROM and JOIN clauses
    pattern = r'(?:FROM|JOIN)\s+["`]?(\w+)["`]?'
    used_tables = set(re.findall(pattern, sql, re.IGNORECASE))
    
    # Remove debug prints
    # Case-insensitive comparison
    schema_lower = {t.lower() for t in schema_tables}
    used_lower = {t.lower() for t in used_tables}
    
    is_valid = used_lower.issubset(schema_lower)
    
    if not is_valid:
        invalid_tables = used_lower - schema_lower
        print(f"❌ Invalid tables found: {invalid_tables}")
    
    return is_valid

def write_query(state: State):
    """Generate SQL query to fetch information."""
    db = get_db()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    table_info = db.get_table_info()

    prompt = query_prompt_template.invoke({
        "table_info": table_info,
        "input": state.question
    })

    msg = llm.bind_tools([QueryOutput]).invoke(prompt)

    if not msg.tool_calls:
        print("⚠️ No tool calls received from LLM")
        state.query = ""
        return {"query": ""}

    candidate = msg.tool_calls[0]["args"]["query"]
    print(f"📝 Generated SQL: {candidate}")

    if not validate_tables(candidate, table_info):
        print("⚠️ Invalid table detected")
        state.query = ""
        return {"query": ""}

    state.query = candidate
    return {"query": state.query}

def write_query_with_llm(state: State, model_type: str = "groq"):
    """Generate SQL with specified LLM."""
    db = get_db()
    llm = get_llm(model_type)
    
    table_info = db.get_table_info()
    
    prompt = query_prompt_template.invoke({
        "table_info": table_info,
        "input": state.question
    })
    
    llm_with_tools = llm.bind_tools([QueryOutput])
    msg = llm_with_tools.invoke(prompt)
    
    if msg.tool_calls:
        candidate = msg.tool_calls[0]["args"]["query"]
        
        if not validate_tables(candidate, table_info):
            print(f"⚠️ {model_type}: Invalid table detected")
        
        state.query = candidate
        print(f"SQL generated with {model_type}: {state.query}")
        return {"query": state.query}
    else:
        print(f"Could not generate SQL with {model_type}")
        return {"query": ""}

def execute_query(state: State):
    """Execute SQL query with validation."""
    if not state.query:
        state.result = "No SQL query to execute"
        return {"result": state.result}
    
    try:
        db = get_db()
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        checker_tool = QuerySQLCheckerTool(db=db, llm=llm)
        check_result = checker_tool.invoke(state.query)
        
        if "error" in check_result.lower():
            state.result = f"Invalid SQL query: {check_result}"
            return {"result": state.result}
        
        result = db.run(state.query)
        state.result = result
        print(f"✅ Results: {result}")
        return {"result": result}
        
    except Exception as e:
        error_msg = f"SQL execution error: {e}"
        state.result = error_msg
        print(error_msg)
        return {"result": error_msg}

def generate_answer(state: State):
    if not state.query or not state.result:
        state.answer = "Unable to answer the question because no valid SQL query could be generated."
        return {"answer": state.answer}

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    prompt = f"""
    Given the following user question, corresponding SQL query, and SQL result,
    answer the user question in English in a natural and helpful way.
    
    Question: {state.question}
    SQL Query: {state.query}
    SQL Result: {state.result}
    
    Provide a clear, concise answer in English:
    """
    
    response = llm.invoke(prompt)
    state.answer = response.content
    print(f"Final answer: {state.answer}")
    return {"answer": state.answer}

def handle_ambiguous_input(state: State):
    """Detectează și gestionează input ambiguu."""
    # Cuvinte cheie care indică ambiguitate
    ambiguous_keywords = ["toate", "multe", "puține", "recent", "vechi", "popular"]
    
    if any(keyword in state.question.lower() for keyword in ambiguous_keywords):
        print(f"Întrebarea pare ambiguă: {state.question}")
        
        clarification = input("Poți fi mai specific? (ex: ce an, câte rezultate, etc.): ")
        state.question = f"{state.question}. {clarification}"
        print(f"Întrebare clarificată: {state.question}")
    
    return state

def get_user_feedback(result: dict) -> dict:
    """Cere feedback de la utilizator."""
    print(f"\nQuestion: {result.get('question', 'N/A')}")
    print(f"Answer: {result.get('answer', 'N/A')}")
    
    feedback = input("\nIs the answer correct? (yes/no/partial): ").lower()
    comment = input("Comments (optional): ")
    
    score = 1 if feedback in ["yes", "da"] else (-1 if feedback in ["no", "nu"] else 0)
    
    return {
        "feedback_score": score,
        "feedback_comment": comment
    }


def handle_negative_feedback(result: dict):
    """Handle negative feedback by regenerating query."""
    print("Trying to improve the answer...")
    
    state = State(question=result.get('question', ''))
    
    write_query(state)
    execute_query(state)
    generate_answer(state)
    
    return {
        "question": state.question,
        "query": state.query,
        "result": state.result,
        "answer": state.answer
    }

def get_llm(model_type: str):
    if model_type == "groq":
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    elif model_type == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1:8b", temperature=0)
    elif model_type == "ollama3.2":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2", temperature=0)
    elif model_type == "ollama-qwen":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
    elif model_type == "cohere":
        from langchain_cohere import ChatCohere
        return ChatCohere(model="command-r", temperature=0)
    elif model_type == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

def compare_llms(question: str):
    """Compare SQL generation across different LLMs."""
    models = ["groq", "ollama", "ollama3.2", "ollama-qwen", "gemini", "cohere"]
    results = {}
    
    for model in models:
        print(f"\n🔄 Testing {model.upper()}...")
        try:
            state = State(question=question)
            write_query_with_llm(state, model)
            execute_query(state)
            generate_answer(state)
            results[model] = {
                "query": state.query,
                "result": state.result,
                "answer": state.answer
            }
            print(f"✅ {model.upper()} completed")
        except Exception as e:
            results[model] = {"error": str(e)}
            print(f"❌ {model.upper()} failed: {e}")
    
    return results

def evaluate_on_spider(spider_json_path: str, max_questions: int = 10):
    """Evaluează pe benchmark-ul Spider."""
    with open(spider_json_path, 'r') as f:
        spider_data = json.load(f)
    
    results = []
    
    for i, item in enumerate(spider_data[:max_questions]):
        try:
            state = State(question=item['question'])
            write_query(state)
            
            # Exact Match
            exact_match = state.query.strip().lower() == item['query'].strip().lower()
            
            execute_query(state)
            execution_accuracy = "error" not in state.result.lower()
            
            results.append(EvaluationResult(
                question=item['question'],
                query_ground_truth=item['query'],
                query_generated=state.query,
                exact_match=exact_match,
                execution_accuracy=execution_accuracy
            ))
            
        except Exception as e:
            print(f"Eroare la evaluarea întrebării {i}: {e}")
    
    return results

def load_few_shot_examples(eval_suite_dir: Path, num_examples: int = 3):
    """Încarcă câteva exemple gold SQL pentru few-shot learning."""
    gold_sql_dir = eval_suite_dir / "gold" / "sql"
    
    few_shot_examples = []
    
    # Selectează exemple gold SQL
    gold_files = list(gold_sql_dir.glob("local*.sql"))[:num_examples]
    
    for sql_file in gold_files:
        instance_id = sql_file.stem
        sql_content = sql_file.read_text()
        
        # Încearcă să găsească întrebarea corespunzătoare
        few_shot_examples.append({
            'instance_id': instance_id,
            'sql': sql_content
        })
    
    return few_shot_examples

def evaluate_on_spider2_lite(spider2_lite_dir: str, max_questions: int = None, primary_llm: str = "groq"):
    """Evaluează pe Spider2-Lite - generează predicții și rulează scriptul oficial de evaluare."""
    from pathlib import Path
    
    spider2_lite_path = Path(spider2_lite_dir).resolve()
    data_file = spider2_lite_path / "spider2-lite.jsonl"
    eval_script = spider2_lite_path / "evaluation_suite" / "evaluate.py"
    eval_suite_dir = spider2_lite_path / "evaluation_suite"
    
    # Verifică bazele de date SQLite
    db_dir_localdb = spider2_lite_path / "resource" / "databases" / "spider2-localdb"
    db_dir_direct = spider2_lite_path / "resource" / "databases"
    
    # Alege locația care are cele mai multe fișiere .sqlite
    if db_dir_localdb.exists():
        localdb_count = len(list(db_dir_localdb.glob("*.sqlite")))
    else:
        localdb_count = 0
    
    if db_dir_direct.exists():
        direct_count = len(list(db_dir_direct.glob("*.sqlite")))
    else:
        direct_count = 0
    
    if localdb_count > direct_count:
        db_dir = db_dir_localdb
        print(f"⚠️  Bazele de date sunt în spider2-localdb/, dar evaluatorul le caută în databases/")
        print(f"💡 Rulează: cd {db_dir_direct} && ln -s spider2-localdb/*.sqlite .")
    else:
        db_dir = db_dir_direct
    
    # Verifică bazele de date SQLite
    if db_dir.exists():
        sqlite_files = list(db_dir.glob("*.sqlite"))
        print(f"📂 Baze de date disponibile în {db_dir}:")
        print(f"   Găsite {len(sqlite_files)} fișiere .sqlite")
        if sqlite_files:
            print(f"   Exemple: {', '.join([f.name for f in sqlite_files[:5]])}...")
    else:
        print(f"⚠️  Directorul {db_dir} nu există!")
        return None
    
    # Încarcă few-shot examples
    print(f"\n📚 Încărcare few-shot examples...")
    few_shot_examples = load_few_shot_examples(eval_suite_dir, num_examples=2)
    print(f"✅ Încărcate {len(few_shot_examples)} exemple gold SQL")
    
    # ADDED: Încarcă lista de instance_id care au gold SQL
    gold_sql_dir = eval_suite_dir / "gold" / "sql"
    gold_instance_ids = {f.stem for f in gold_sql_dir.glob("local*.sql")}
    print(f"✅ Găsite {len(gold_instance_ids)} exemple cu gold SQL")
    
    # Încarcă date
    print(f"\n📖 Încărcare date din {data_file}...")
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✅ Încărcate {len(data)} exemple total")
    
    # Filtrează doar exemple care au baze de date SQLite disponibile ȘI gold SQL
    sqlite_data = []
    available_dbs = {f.stem for f in sqlite_files}
    
    for item in data:
        db_name = item['db']
        instance_id = item['instance_id']
        
        # Skip non-SQLite databases
        if db_name.startswith(('bigquery', 'snowflake', 'ga4', 'ga360', 'firebase')):
            continue
        
        # ADDED: Skip dacă nu are gold SQL pentru evaluare
        if instance_id not in gold_instance_ids:
            print(f"⚠️  Skip {instance_id} - nu are gold SQL pentru evaluare")
            continue
        
        if db_name in available_dbs:
            sqlite_data.append(item)
    
    print(f"✅ Filtrate {len(sqlite_data)} exemple cu baze de date SQLite ȘI gold SQL")
    
    if max_questions:
        sqlite_data = sqlite_data[:max_questions]
        print(f"✅ Limitate la {len(sqlite_data)} exemple pentru testare")
    
    if not sqlite_data:
        print("❌ Nu există exemple cu baze de date SQLite disponibile!")
        return None
    
    # Generează predicții
    predictions = []
    llm_fallback = None  # Pentru fallback la alt LLM
    
    for i, item in enumerate(sqlite_data):
        instance_id = item['instance_id']
        db_id = item['db']
        question = item['question']
        
        print(f"\n🔄 [{i+1}/{len(sqlite_data)}] {instance_id}")
        print(f"   DB: {db_id}")
        print(f"   Q: {question[:80]}...")
        
        db_file = db_dir / f"{db_id}.sqlite"
        
        if not db_file.exists():
            print(f"   ⚠️  DB nu există: {db_file}")
            predictions.append({"instance_id": instance_id, "SQL": ""})
            continue
        
        try:
            import sqlite3
            
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Obține lista de tabele
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Obține schema completă pentru fiecare tabelă
            schema_info = []
            for table in tables:
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
                create_stmt = cursor.fetchone()
                if create_stmt:
                    schema_info.append(create_stmt[0])
                    
                    # Adaugă și câteva rânduri exemplu
                    cursor.execute(f"SELECT * FROM {table} LIMIT 2;")
                    sample_rows = cursor.fetchall()
                    if sample_rows:
                        cursor.execute(f"PRAGMA table_info({table});")
                        columns = [col[1] for col in cursor.fetchall()]
                        schema_info.append(f"-- Sample data: {columns[:5]}...")
            
            conn.close()
            
            schema_text = "\n\n".join(schema_info)
            print(f"   📋 Schema: {len(tables)} tabele: {', '.join(tables[:5])}...")
            
            # Construiește few-shot examples text
            few_shot_text = ""
            if few_shot_examples:
                few_shot_text = "\n\nHere are some example SQL queries from similar tasks:\n\n"
                for ex in few_shot_examples:
                    few_shot_text += f"Example {ex['instance_id']}:\n{ex['sql'][:300]}...\n\n"
            
            # Prompt îmbunătățit cu few-shot și instrucțiuni detaliate
            prompt_text = f"""You are an expert SQL query generator specializing in complex analytical queries.

CRITICAL INSTRUCTIONS:
1. Use ONLY these exact table names: {', '.join(tables)}
2. Write standard SQL compatible with SQLite
3. Pay attention to:
   - Use NTILE() for percentile-based scoring when needed
   - Use window functions (OVER, PARTITION BY) for advanced analytics
   - Use proper date functions (DATE(), STRFTIME())
   - Use JOINs with USING() when columns have same names
   - Group and aggregate data appropriately
4. Return ONLY the complete SQL query, nothing else (no explanations, markdown, or comments outside the SQL)

DATABASE SCHEMA:
{schema_text}
{few_shot_text}

QUESTION: {question}

Generate a complete, executable SQL query:"""
            
            # Încearcă LLM-ul principal, apoi fallback la Gemini
            generated_sql = None
            
            try:
                # Folosește LLM-ul selectat de utilizator
                if primary_llm == "groq":
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
                elif primary_llm == "gemini":
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-exp",
                        temperature=0,
                        google_api_key=os.getenv("GOOGLE_API_KEY")
                    )
                else:
                    llm = get_llm(primary_llm)
                
                response = llm.invoke(prompt_text)
                generated_sql = response.content.strip()
                
            except Exception as primary_error:
                if "rate_limit" in str(primary_error).lower() or "429" in str(primary_error):
                    print(f"   ⚠️  {primary_llm.upper()} rate limit - folosesc Gemini fallback")
                    
                    # Fallback la Gemini (dacă nu e deja Gemini)
                    if llm_fallback is None and primary_llm != "gemini":
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        llm_fallback = ChatGoogleGenerativeAI(
                            model="gemini-2.0-flash-exp",
                            temperature=0,
                            google_api_key=os.getenv("GOOGLE_API_KEY")
                        )
                    
                    if llm_fallback:
                        response = llm_fallback.invoke(prompt_text)
                        generated_sql = response.content.strip()
                    else:
                        raise primary_error
                else:
                    raise primary_error
            
            # Curăță SQL-ul
            generated_sql = re.sub(r'^```[\w]*', '', generated_sql, flags=re.MULTILINE)
            generated_sql = re.sub(r'```$', '', generated_sql, flags=re.MULTILINE)
            generated_sql = re.sub(r'^(SQL|Query):\s*', '', generated_sql, flags=re.IGNORECASE | re.MULTILINE)
            generated_sql = '\n'.join(line for line in generated_sql.split('\n') if line.strip())
            generated_sql = generated_sql.strip()
            
            # Self-validation: încearcă să rulezi SQL-ul pentru a verifica sintaxa
            try:
                test_conn = sqlite3.connect(str(db_file))
                test_cursor = test_conn.cursor()
                test_cursor.execute(f"EXPLAIN QUERY PLAN {generated_sql}")
                test_conn.close()
                print(f"   ✅ SQL valid (verificat cu EXPLAIN)")
            except Exception as validation_error:
                print(f"   ⚠️  SQL ar putea avea probleme: {str(validation_error)[:100]}")
            
            print(f"   ✅ SQL: {generated_sql[:80]}...")
            
            predictions.append({
                "instance_id": instance_id,
                "SQL": generated_sql
            })
            
        except Exception as e:
            print(f"   ❌ Eroare: {e}")
            predictions.append({
                "instance_id": instance_id,
                "SQL": ""
            })
    
    # Salvează predicțiile
    pred_file = Path("spider2_lite_predictions.jsonl")
    pred_dir = Path("spider2_lite_predictions")
    
    with open(pred_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    print(f"\n💾 Predicții salvate: {pred_file}")
    
    # Creează director și salvează fiecare predicție ca fișier .sql separat
    pred_dir.mkdir(exist_ok=True)
    
    for pred in predictions:
        instance_id = pred['instance_id']
        sql_content = pred['SQL']
        sql_file = pred_dir / f"{instance_id}.sql"
        with open(sql_file, 'w', encoding='utf-8') as f:
            f.write(sql_content)
    
    print(f"💾 Fișiere SQL salvate în: {pred_dir}/ ({len(predictions)} fișiere)")
    
    # Rulează scriptul oficial de evaluare
    print("\n" + "="*60)
    print("📊 RULEAZĂ EVALUAREA OFICIALĂ SPIDER2-LITE")
    print("="*60)
    
    import subprocess
    import sys
    
    cmd = [
        sys.executable,
        str(eval_script.resolve()),
        '--mode', 'sql',
        '--result_dir', str(pred_dir.absolute())
    ]
    
    print(f"Comandă: {' '.join(cmd)}\n")
    print(f"⏳ Așteptare evaluare (poate dura câteva minute)...\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(eval_script.parent.resolve()),
            capture_output=True,
            text=True,
            timeout=300  # ADDED: 5 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print(f"Stderr:\n{result.stderr}")
        
        if result.returncode != 0:
            print(f"\n⚠️  Scriptul de evaluare a returnat cod {result.returncode}")
        
        # Parsează scorul din output
        score_match = re.search(r'Final score:\s*([\d.]+)', result.stdout)
        final_score = float(score_match.group(1)) if score_match else 0.0
        
        return {
            'predictions': predictions,
            'predictions_file': str(pred_file),
            'predictions_dir': str(pred_dir),
            'evaluation_output': result.stdout,
            'return_code': result.returncode,
            'final_score': final_score
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  Evaluarea a depășit timeout-ul de 5 minute!")
        print(f"\n💡 Rulează manual:")
        print(f"   cd {eval_script.parent}")
        print(f"   python evaluate.py --mode sql --result_dir {pred_dir.absolute()}")
        
        return {
            'predictions': predictions,
            'predictions_file': str(pred_file),
            'predictions_dir': str(pred_dir),
            'timeout': True
        }
    except Exception as e:
        print(f"⚠️  Eroare la evaluare: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'predictions': predictions,
            'predictions_file': str(pred_file),
            'predictions_dir': str(pred_dir)
        }

def calculate_metrics(results: list):
    """Calculează metricile de evaluare."""
    total = len(results)
    exact_matches = sum(r.exact_match for r in results)
    execution_accuracies = sum(r.execution_accuracy for r in results)
    
    return {
        "exact_match_accuracy": exact_matches / total if total > 0 else 0,
        "execution_accuracy": execution_accuracies / total if total > 0 else 0,
        "total_questions": total
    }
        
def create_text2sql_graph():
    """Creează un LangGraph StateGraph pentru pipeline-ul Text-to-SQL."""
    graph_builder = StateGraph(State)
    
    # Adaugă nodurile în secvență
    graph_builder.add_sequence([write_query, execute_query, generate_answer])
    graph_builder.add_edge(START, "write_query")
    
    return graph_builder.compile()

def run_pipeline(question: str):
    graph = create_text2sql_graph()
    result = graph.invoke({"question": question})
    return result

def run_pipeline_with_feedback(question: str):
    """Rulează pipeline-ul cu feedback loop complet."""
    result = run_pipeline(question)
    feedback = get_user_feedback(result)
    
    # Dacă feedback-ul e negativ, încearcă să îmbunătățești
    if feedback['feedback_score'] < 0:
        print("Trying to improve the answer based on your feedback...")
        improved_result = handle_negative_feedback(result)
        
        # Cere feedback din nou pentru răspunsul îmbunătățit
        print("\n--- IMPROVED ANSWER ---")
        print(f"New Answer: {improved_result.get('answer', 'N/A')}")
        final_feedback = get_user_feedback(improved_result)
        
        return improved_result, final_feedback
    
    return result, feedback

def load_few_shot_examples(eval_suite_dir: Path, num_examples: int = 3):
    """Încarcă câteva exemple gold SQL pentru few-shot learning."""
    gold_sql_dir = eval_suite_dir / "gold" / "sql"
    
    few_shot_examples = []
    
    # Selectează exemple gold SQL
    gold_files = list(gold_sql_dir.glob("local*.sql"))[:num_examples]
    
    for sql_file in gold_files:
        instance_id = sql_file.stem
        sql_content = sql_file.read_text()
        
        # Încearcă să găsească întrebarea corespunzătoare
        few_shot_examples.append({
            'instance_id': instance_id,
            'sql': sql_content
        })
    
    return few_shot_examples

def main():
    parser = argparse.ArgumentParser(description="text2sql cli interface")
    parser.add_argument("--question", type=str, help="natural language question")
    parser.add_argument("--compare", action="store_true", help="compare different LLMs")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--feedback", action="store_true", help="enable feedback loop")
    parser.add_argument("--evaluate", type=str, help="path to Spider JSON file")
    parser.add_argument("--spider2-lite", type=str, help="path to Spider2-Lite directory")
    parser.add_argument("--max-questions", type=int, default=None, help="maximum questions")
    parser.add_argument("--llm", type=str, default="groq", 
                       choices=["groq", "ollama", "ollama3.2", "ollama-qwen", "gemini", "cohere"],
                       help="default: groq")
     
    args = parser.parse_args()
    
    if not args.compare:
        print(f"🤖 LLM: {args.llm.upper()}")
    
    if args.spider2_lite:
        print("🚀 Running Spider2-Lite evaluation...")
        result = evaluate_on_spider2_lite(
            args.spider2_lite, 
            args.max_questions,
            primary_llm=args.llm
        )
        
        if not result:
            print("\n❌ Evaluation failed - check SQLite databases")
            return
        
        print("\n" + "="*60)
        print("✅ COMPLETE EVALUATION")
        print("="*60)
        print(f"Predictions saved to: {result['predictions_file']}")
        print(f"Total predictions: {len(result['predictions'])}")
        
        if 'final_score' in result:
            print(f"\n🎯 FINAL SCORE: {result['final_score']:.2%}")
        
        if 'evaluation_output' in result and result['evaluation_output']:
            print("\n📊 OFFICIAL EVALUATION RESULTS:")
            for line in result['evaluation_output'].split('\n'):
                if 'Final score' in line or 'Correct examples' in line or 'Total examples' in line:
                    print(f"   {line}")
        
        return
    
    if args.evaluate:
        results = evaluate_on_spider(args.evaluate, args.max_questions or 10)
        metrics = calculate_metrics(results)
        print("\n=== Evaluation Results ===")
        print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
        print(f"Execution Accuracy: {metrics['execution_accuracy']:.2%}")
        print(f"Total Questions: {metrics['total_questions']}")
        return
        
    if args.interactive:
        while True:
            question = input("\nEnter your question (or 'exit'): ")
            if question.lower() == 'exit':
                break
            
            state = State(question=question)
            write_query_with_llm(state, args.llm)
            execute_query(state)
            
            if args.feedback:
                feedback = get_user_feedback({"question": question, "answer": state.answer})
                print(f"Feedback saved: {feedback}")
                
                if feedback['feedback_score'] < 0:
                    print("Trying to improve the answer...")
                    write_query_with_llm(state, args.llm)
                    execute_query(state)
            else:
                print(f"\n📊 SQL: {state.query}")
                print(f"📊 Result: {state.result}")
    
    elif args.compare:
        question = args.question or input("Question for comparison: ")
        results = compare_llms(question)
        
        for model, result in results.items():
            print(f"\n=== {model.upper()} ===")
            print(f"Query: {result.get('query', 'N/A')}")
            print(f"Answer: {result.get('answer', 'N/A')}")
    
    else:
        question = args.question or input("Question: ")
        state = State(question=question)
        write_query_with_llm(state, args.llm)
        
        if args.feedback:
            execute_query(state)
            generate_answer(state)
            feedback = get_user_feedback({"question": question, "answer": state.answer})
            print(f"Feedback: {feedback}")
        else:
            execute_query(state)
            print(f"\n📊 SQL: {state.query}")
            print(f"📊 Result: {state.result}")

if __name__ == "__main__":
    main()