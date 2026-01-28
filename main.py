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
import sqlite3
import random

load_dotenv()

# sql lite local, nu am credidentiale pt google cloud(bigquery) sau snowflake(sf_bq)
# 100k tokens pe grok, 1 cuvant = 1,4 * tokens

# ============================================================================
# 1. DATA MODELS
# ============================================================================
# fiecare nod din langGraph primeste un state si l modifica
class State(BaseModel):
    question: str  #intrebarea intiala
    clarified_question: str = "" #intrebarea clarificata
    is_ambiguous: bool = False 
    chain_of_thought: str = "" #analiza cot
    query: str = ""#query generat
    result: str = ""# rezultatul query ului
    answer: str = "" #raspunsul final
    llm_used: str = "groq" # ce llm am folosit

# fortam llm ul sa returneze doar query sql, elimina halucinatiile, nu da texte random
class QueryOutput(BaseModel):
    """Generated SQL query."""
    query: str = Field(description="Syntactically valid SQL query.")

class EvaluationResult(BaseModel):
    question: str #intrebarea din benckmark
    query_ground_truth: str # query ul corect din spider
    query_generated: str # query ul generat de model
    exact_match: bool 
    execution_accuracy: bool # daca query ul generat ruleaza fara erori

# ============================================================================
# 2. PROMPT TEMPLATES 
# ============================================================================
# clasa pt centralizarea prompturilor folosite in pipeline
class PromptManager:    
    SYSTEM_SCHEMA = """
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

DATE HANDLING:
- Check sample data to understand the actual date format
- If dates are stored as Unix timestamps (large integers), use appropriate conversion
- For date comparisons, use STRFTIME or date functions as needed
- Example: WHERE STRFTIME('%Y', datetime(release_date/1000, 'unixepoch')) = '2024'

Database schema (with sample rows):
{table_info}

Sample data from relevant tables:
{sample_data}
"""

    SYSTEM_COT = """
You are an expert SQL analyst. Your task is to think through complex questions step-by-step.

For the given question, provide a Chain-of-Thought analysis:
1. What tables might be involved?
2. What columns would you need?
3. What joins are necessary?
4. What filters or aggregations are needed?
5. What is the final query structure?

Then provide the final SQL query.

Database schema:
{table_info}
"""

    SYSTEM_AMBIGUITY = """
Analyze the following question and determine if it is ambiguous or vague.
Identify which aspects are missing and suggest clarification questions.

Respond in JSON format:
{{
    "is_ambiguous": true/false,
    "ambiguity_reasons": ["list of reasons"],
    "clarification_questions": ["question 1", "question 2"]
}}
"""

    SYSTEM_SUMMARIZE = """
Given the following user question, corresponding SQL query, and SQL result,
answer the user question in a natural and helpful way.

Be concise, clear, and focus on answering what was asked.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Provide a clear, concise answer in English:
"""
# builder patern, construim chatpromttemplate din langchain
    @staticmethod
    def get_schema_prompt():
        return ChatPromptTemplate([
            ("system", PromptManager.SYSTEM_SCHEMA),
            ("user", "Question: {input}")
        ])

    @staticmethod
    def get_cot_prompt():
        return ChatPromptTemplate([
            ("system", PromptManager.SYSTEM_COT),
            ("user", "Question: {input}")
        ])

    @staticmethod
    def get_ambiguity_prompt():
        return ChatPromptTemplate([
            ("system", PromptManager.SYSTEM_AMBIGUITY),
            ("user", "Intrebare: {question}")
        ])

    @staticmethod
    def get_summarize_prompt():
        return ChatPromptTemplate([
            ("system", PromptManager.SYSTEM_SUMMARIZE)
        ])

# ============================================================================
# 3. DATABASE UTILITIES
# ============================================================================

# cautam baza de date in folderul db, si returnam un sqldatabase object, nu o folosesc dar o pastrez ca model
def get_db():
    """Initialize SQLite database connection."""
    from sqlalchemy.pool import StaticPool
    from sqlalchemy import event
    
    db_path = os.path.join(os.path.dirname(__file__), "db", "netflixdb.sqlite")
    
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={'check_same_thread': False}, # acces multithreaded la sqlite
        poolclass=StaticPool
    )
    
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA parse_time=0")
        cursor.close()
    
    return SQLDatabase(engine, sample_rows_in_table_info=0)

# extragem cateva linii de date din tabelele cheie pentru a le afisa in prompt
def get_sample_data(db_name: str, table_name: str, limit: int = 3) -> str:
    
    try:
        db_path = os.path.join(os.path.dirname(__file__), "db", f"{db_name}.sqlite")
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get sample rows
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        rows = cursor.fetchall()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        
        conn.close()
        
        if not rows:
            return f"No sample data available for {table_name}"
        
        # Format as readable table
        result = f"\nSample data from {table_name}:\n"
        result += " | ".join(columns) + "\n"
        result += "-" * (len(" | ".join(columns))) + "\n"
        
        for row in rows:
            result += " | ".join(str(val) if val is not None else "NULL" for val in row) + "\n"
        
        return result
        
    except Exception as e:
        return f"Error getting sample data: {str(e)}"
# extragem sample data din fiecare tabel, si concatenam rezultatele intr un singur string
def get_db_with_samples():
    from sqlalchemy.pool import StaticPool
    from sqlalchemy import event
    
    db_path = os.path.join(os.path.dirname(__file__), "db", "netflixdb.sqlite")
    
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={'check_same_thread': False},
        poolclass=StaticPool
    )
    
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA parse_time=0")
        cursor.close()
    
    db = SQLDatabase(engine, sample_rows_in_table_info=0)
    
    # Add sample data for key tables
    sample_data = ""
    try:
        tables = ["movie", "series", "episode"]  # Common table names
        for table in tables:
            sample_data += get_sample_data("netflixdb", table)
    except:
        # Fallback if table names are different
        pass
    
    # Store sample data in db object for later use
    db._sample_data = sample_data
    return db

# extragem numele tabelelor din schema
def get_schema_tables(table_info: str):
    """Extract table names from schema info."""
    tables = set(re.findall(r'CREATE TABLE ["\']?(\w+)["\']?', table_info, re.IGNORECASE))
    tables.update(re.findall(r'Table:\s*["\']?(\w+)["\']?', table_info, re.IGNORECASE))
    return tables

# verificam daca tabelele folosite in query exista in schema
def validate_tables(sql: str, table_info: str):
    schema_tables = get_schema_tables(table_info)
    pattern = r'(?:FROM|JOIN)\s+["`]?(\w+)["`]?'
    used_tables = set(re.findall(pattern, sql, re.IGNORECASE))
    
    schema_lower = {t.lower() for t in schema_tables}
    used_lower = {t.lower() for t in used_tables}
    
    is_valid = used_lower.issubset(schema_lower)
    
    if not is_valid:
        invalid_tables = used_lower - schema_lower
        print(f"Invalid tables found: {invalid_tables}")
    
    return is_valid

# verificam sintaxa sql folosind sqlite3
def validate_sql_syntax(sql: str, db_file: str) -> tuple[bool, str]:
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        conn.close()
        return True, "SQL syntax valid"
    except Exception as e:
        return False, str(e)

# ============================================================================
# 4. AMBIGUITY DETECTION (NODE 1)
# ============================================================================
# daca intrebarea este deja ambigua, dam skip, alfet facem o instanta 
# cu groq deterministic (temperature 0) care analizeaza intrebarea si returneaza un json

def detect_ambiguity(state: State):
    if state.is_ambiguous:
        return state
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    prompt = PromptManager.get_ambiguity_prompt().invoke({
        "question": state.question
    })
    
    try:
        response = llm.invoke(prompt)
        analysis = json.loads(response.content)
        
        if analysis.get("is_ambiguous", False):
            print(f"\n[Warning] Ambiguous question detected!")
            print(f"Reasons: {', '.join(analysis.get('ambiguity_reasons', []))}")
            
            state.is_ambiguous = True
            
            print("\nClarification questions:")
            for i, q in enumerate(analysis.get('clarification_questions', []), 1):
                print(f"  {i}. {q}")
            
            clarification = input("\nProvide additional context (or press Enter to continue): ").strip()
            
            if clarification:
                state.clarified_question = f"{state.question}. {clarification}"
            else:
                state.clarified_question = state.question
        else:
            state.clarified_question = state.question
    
    except json.JSONDecodeError:
        print("[Warning] Could not parse ambiguity analysis")
        state.clarified_question = state.question
    except Exception as e:
        print(f"[Warning] Ambiguity detection error: {e}")
        state.clarified_question = state.question
    
    return state

# ============================================================================
# 5. CHAIN-OF-THOUGHT REASONING (NODE 2)
# ============================================================================
# generam analiza cot folosind groq, imparte intrebarea in pasi logici
def generate_chain_of_thought(state: State):
    if not state.clarified_question:
        state.clarified_question = state.question
    
    db = get_db_with_samples()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    table_info = db.get_table_info()
    sample_data = getattr(db, '_sample_data', '')
    
    prompt = PromptManager.get_cot_prompt().invoke({
        "table_info": table_info + "\n" + sample_data,
        "input": state.clarified_question
    })
    
    response = llm.invoke(prompt)
    state.chain_of_thought = response.content
    
    print(f"\n[CoT] Chain-of-Thought Analysis:")
    print("-" * 60)
    print(state.chain_of_thought[:500] + "..." if len(state.chain_of_thought) > 500 else state.chain_of_thought)
    print("-" * 60)
    
    return state

# ============================================================================
# 6. SQL GENERATION (NODE 3)
# ============================================================================

# generam sql folosind groq, 
def write_query(state: State):
    if not state.clarified_question:
        state.clarified_question = state.question
    
    db = get_db_with_samples()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    table_info = db.get_table_info()
    sample_data = getattr(db, '_sample_data', '')
    
    # Include CoT in prompt if available
    system_msg = PromptManager.SYSTEM_SCHEMA
    if state.chain_of_thought:
        system_msg += f"\n\nPrevious analysis:\n{state.chain_of_thought}\n\nNow generate the final SQL query:"
    
    prompt = ChatPromptTemplate([
        ("system", system_msg),
        ("user", "Question: {input}")
    ]).invoke({
        "table_info": table_info,
        "sample_data": sample_data,
        "input": state.clarified_question
    })
    
    llm_with_tools = llm.bind_tools([QueryOutput])
    msg = llm_with_tools.invoke(prompt)
    
    if not msg.tool_calls:
        print("⚠️  No SQL generated from LLM")
        state.query = ""
        return state
    
    candidate = msg.tool_calls[0]["args"]["query"]
    print(f"\nSQL generated: {candidate}")
    
    if not validate_tables(candidate, table_info):
        print("[Warning] Invalid table detected")
        state.query = ""
        return state
    
    state.query = candidate
    state.llm_used = "groq"
    return state

# aceasi functie dar cu suport pentru mai multe llm uri
# gemini si cohere necesita parsare text mai complexa
def write_query_with_llm(state: State, model_type: str = "groq"):
    """Generate SQL with specified LLM."""
    if not state.clarified_question:
        state.clarified_question = state.question
    
    db = get_db_with_samples()
    llm = get_llm(model_type)
    
    table_info = db.get_table_info()
    sample_data = getattr(db, '_sample_data', '')
    
    system_msg = PromptManager.SYSTEM_SCHEMA
    if state.chain_of_thought:
        system_msg += f"\n\nPrevious analysis:\n{state.chain_of_thought}\n\nNow generate the final SQL query:"
    
    prompt = ChatPromptTemplate([
        ("system", system_msg),
        ("user", "Question: {input}")
    ]).invoke({
        "table_info": table_info,
        "sample_data": sample_data,
        "input": state.clarified_question
    })
    
    if model_type in ["groq", "ollama", "ollama3.2", "ollama-qwen"]:
        llm_with_tools = llm.bind_tools([QueryOutput])
        msg = llm_with_tools.invoke(prompt)
        
        if msg.tool_calls:
            candidate = msg.tool_calls[0]["args"]["query"]
            
            if not validate_tables(candidate, table_info):
                print(f"[Warning] {model_type}: Invalid table detected")
            
            state.query = candidate
            state.llm_used = model_type
            print(f"[Success] SQL generated with {model_type}: {candidate[:80]}...")
            return state
        else:
            print(f"Could not generate SQL with {model_type}")
            return state
    else:
        # For Gemini & Cohere: use direct text generation and parse
        response = llm.invoke(prompt)
        response_text = response.content
        
        print(f"  Response received (length: {len(response_text)})")
        
        # Enhanced SQL extraction patterns
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?)\s*```',
            r'(SELECT\s+.*?)(?:\n\n|$|\Z)',
            r'(?:^|\n)\s*(SELECT\s+.*?)(?:\n\s*\n|\Z)',
        ]
        
        state.query = ""
        for pattern in sql_patterns:
            sql_match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if sql_match:
                candidate_sql = sql_match.group(1).strip()
                # Remove any leading junk text before SELECT
                lines = candidate_sql.split('\n')
                clean_lines = []
                found_select = False
                for line in lines:
                    if not found_select and 'SELECT' in line.upper():
                        # Find the position of SELECT and take from there
                        select_pos = line.upper().find('SELECT')
                        clean_lines.append(line[select_pos:])
                        found_select = True
                    elif found_select:
                        clean_lines.append(line)
                
                if clean_lines:
                    state.query = '\n'.join(clean_lines).strip()
                    break
        
        # Fallback: look for any line starting with SELECT
        if not state.query:
            lines = response_text.split('\n')
            for line in lines:
                line_clean = line.strip()
                if line_clean.upper().startswith('SELECT'):
                    state.query = line_clean
                    break
        
        if state.query:
            # Final cleanup
            if state.query.endswith(';'):
                state.query = state.query[:-1].strip()
            
            # Remove common prefixes
            state.query = re.sub(r'^(SQL|Query|Answer):\s*', '', state.query, flags=re.IGNORECASE)
            
            print(f"[Success] SQL generated with {model_type}: {state.query[:80]}...")
            state.llm_used = model_type
            return state
        else:
            print(f"Could not extract SQL with {model_type}")
            print(f"  Raw response: {response_text[:200]}...")
            return state

# ============================================================================
# 7. QUERY VALIDATION & EXECUTION (NODE 4)
# ============================================================================

# cath la erorile de sintaxa si de executie sql
# luam sql din state si il rulam pe db ul sqlite
# folosim llm ul pentru a valida sintaxa sql inainte de rulare
def validate_and_execute_query(state: State):
    if not state.query:
        state.result = "No SQL query to execute"
        return state
    
    try:
        db = get_db_with_samples()
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        checker_tool = QuerySQLCheckerTool(db=db, llm=llm)
        check_result = checker_tool.invoke(state.query)
        
        if "error" in check_result.lower():
            state.result = f"Invalid SQL query: {check_result}"
            print(f"Validation failed: {check_result}")
            return state
        
        result = db.run(state.query)
        state.result = result
        print(f"Query executed successfully")
        return state
        
    except Exception as e:
        error_msg = f"SQL execution error: {e}"
        state.result = error_msg
        print(error_msg)
        return state

# ============================================================================
# 8. ANSWER GENERATION (NODE 5)
# ============================================================================

# convertem rezultatul sql in raspuns natural 
def generate_answer(state: State):
    if not state.query or not state.result:
        state.answer = "Unable to answer the question because no valid SQL query could be generated."
        return state
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    prompt = PromptManager.SYSTEM_SUMMARIZE.format(
        question=state.question,
        query=state.query,
        result=state.result
    )
    
    response = llm.invoke(prompt)
    state.answer = response.content
    print(f"\nFinal Answer:\n{state.answer}")
    return state

# ============================================================================
# 9. LLM MANAGEMENT
# ============================================================================
# luam instant a llm ului in functie de modelul selectat
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
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        return ChatCohere(
            model="command-r-plus-08-2024", 
            temperature=0,
            cohere_api_key=cohere_api_key,
            max_tokens=1000
        )
    elif model_type == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            google_api_key=google_api_key
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

# evaluam toate llm urile pe aceeasi intrebare si comparam rezultatele
# construim un promt identic pentru toate llm urile
# executa tot pipeline ul pentru fiecare llm
def compare_llms(question: str):
    models = ["groq", "ollama", "ollama3.2", "ollama-qwen", "gemini", "cohere"]
    results = {}
    
    for model in models:
        print(f"\nTesting {model.upper()}...")
        try:
            if model == "cohere":
                cohere_api_key = os.getenv("COHERE_API_KEY")
                if not cohere_api_key:
                    raise ValueError("COHERE_API_KEY not configured. Skipping Cohere.")
            if model == "gemini" and not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY not configured. Skipping Gemini.")
            
            state = State(question=question)
            state.clarified_question = question
            
            db = get_db_with_samples()
            llm = get_llm(model)
            table_info = db.get_table_info()
            sample_data = getattr(db, '_sample_data', '')
            
            system_msg = PromptManager.SYSTEM_SCHEMA
            prompt = ChatPromptTemplate([
                ("system", system_msg),
                ("user", "Question: {input}")
            ]).invoke({
                "table_info": table_info,
                "sample_data": sample_data,
                "input": state.clarified_question
            })
            
            if model in ["groq", "ollama", "ollama3.2", "ollama-qwen"]:
                llm_with_tools = llm.bind_tools([QueryOutput])
                msg = llm_with_tools.invoke(prompt)
                
                if msg.tool_calls:
                    state.query = msg.tool_calls[0]["args"]["query"]
                    print(f"[Success] SQL generated with {model}: {state.query[:80]}...")
                else:
                    print(f"Could not generate SQL with {model}")
                    state.query = ""
            else:
                # Enhanced parsing for Gemini & Cohere
                response = llm.invoke(prompt)
                response_text = response.content
                
                print(f"  Response received (length: {len(response_text)})")
                
                # Enhanced SQL extraction patterns
                sql_patterns = [
                    r'```sql\s*(.*?)\s*```',
                    r'```\s*(SELECT.*?)\s*```',
                    r'(SELECT\s+.*?)(?:\n\n|$|\Z)',
                    r'(?:^|\n)\s*(SELECT\s+.*?)(?:\n\s*\n|\Z)',
                ]
                
                state.query = ""
                for pattern in sql_patterns:
                    sql_match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                    if sql_match:
                        candidate_sql = sql_match.group(1).strip()
                        # Remove any leading junk text before SELECT
                        lines = candidate_sql.split('\n')
                        clean_lines = []
                        found_select = False
                        for line in lines:
                            if not found_select and 'SELECT' in line.upper():
                                # Find the position of SELECT and take from there
                                select_pos = line.upper().find('SELECT')
                                clean_lines.append(line[select_pos:])
                                found_select = True
                            elif found_select:
                                clean_lines.append(line)
                        
                        if clean_lines:
                            state.query = '\n'.join(clean_lines).strip()
                            break
                
                # Fallback: look for any line starting with SELECT
                if not state.query:
                    lines = response_text.split('\n')
                    for line in lines:
                        line_clean = line.strip()
                        if line_clean.upper().startswith('SELECT'):
                            state.query = line_clean
                            break
                
                if state.query:
                    # Final cleanup
                    if state.query.endswith(';'):
                        state.query = state.query[:-1].strip()
                    
                    # Remove common prefixes
                    state.query = re.sub(r'^(SQL|Query|Answer):\s*', '', state.query, flags=re.IGNORECASE)
                    
                    print(f"[Success] SQL generated with {model}: {state.query[:80]}...")
                else:
                    print(f"Could not extract SQL with {model}")
                    print(f"  Raw response: {response_text[:200]}...")
            
            if state.query:
                validate_and_execute_query(state)
                generate_answer(state)
                results[model] = {
                    "query": state.query,
                    "result": state.result,
                    "answer": state.answer
                }
                print(f"{model.upper()} completed")
            else:
                results[model] = {"error": "No SQL generated"}
                print(f"{model.upper()} failed: No SQL generated")
        except ValueError as ve:
            results[model] = {"error": str(ve)}
            print(f"{model.upper()} skipped: {str(ve)}")
        except Exception as e:
            error_msg = str(e)
            print(f"  Full error: {error_msg}")
            
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                results[model] = {"error": "Rate limit exceeded"}
                print(f"{model.upper()} failed: Rate limit")
            elif "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg.lower():
                results[model] = {"error": "API Key invalid or unauthorized"}
                print(f"{model.upper()} failed: Check API Key")
            elif "400" in error_msg or "bad request" in error_msg.lower():
                results[model] = {"error": f"Bad request: {error_msg[:100]}"}
                print(f"{model.upper()} failed: Bad request")
            elif "timeout" in error_msg.lower():
                results[model] = {"error": "Request timeout"}
                print(f"{model.upper()} failed: Timeout")
            else:
                results[model] = {"error": error_msg[:200]}
                print(f"{model.upper()} failed: {error_msg[:100]}")
    
    return results

# ============================================================================
# 11. EVALUATION & BENCHMARKING
# ============================================================================

# evaluam modelul pe benchmark ul spider
# comparam query ul generat cu cel corect din spider
# exact match daca sql generat == sql din local character by character
def evaluate_on_spider(spider_json_path: str, max_questions: int = 10):
    
    with open(spider_json_path, 'r') as f:
        spider_data = json.load(f)
    
    results = []
    
    for i, item in enumerate(spider_data[:max_questions]):
        try:
            state = State(question=item['question'])
            state.clarified_question = item['question']
            write_query(state)
            
            exact_match = state.query.strip().lower() == item['query'].strip().lower()
            
            validate_and_execute_query(state)
            execution_accuracy = "error" not in state.result.lower()
            
            results.append(EvaluationResult(
                question=item['question'],
                query_ground_truth=item['query'],
                query_generated=state.query,
                exact_match=exact_match,
                execution_accuracy=execution_accuracy
            ))
            
        except Exception as e:
            print(f"Error evaluating question {i}: {e}")
    
    return results
# calculam metricele de performanta
# count la rezultate
def calculate_metrics(results: list):
    total = len(results)
    exact_matches = sum(r.exact_match for r in results)
    execution_accuracies = sum(r.execution_accuracy for r in results)
    
    return {
        "exact_match_accuracy": exact_matches / total if total > 0 else 0,
        "execution_accuracy": execution_accuracies / total if total > 0 else 0,
        "total_questions": total
    }

# evaluam modelul pe spider2-lite
def evaluate_on_spider2_lite(spider2_lite_dir: str, max_questions: int = None, primary_llm: str = "groq"):
    from pathlib import Path
    
    #cai absolute catre fisierele necesare
    spider2_lite_path = Path(spider2_lite_dir).resolve()
    data_file = spider2_lite_path / "spider2-lite.jsonl"
    eval_script = spider2_lite_path / "evaluation_suite" / "evaluate.py"
    eval_suite_dir = spider2_lite_path / "evaluation_suite"
    
    db_dir_localdb = spider2_lite_path / "resource" / "databases" / "spider2-localdb"
    db_dir_direct = spider2_lite_path / "resource" / "databases"
    
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
        print(f" Databases are in spider2-localdb/, but the evaluator looks in databases/")
        print(f"Run: cd {db_dir_direct} && ln -s spider2-localdb/*.sqlite .")
    else:
        db_dir = db_dir_direct
    
    if db_dir.exists():
        sqlite_files = list(db_dir.glob("*.sqlite"))
        print(f"Found {len(sqlite_files)} SQLite databases")
    else:
        print(f"Directory {db_dir} does not exist!")
        return None
    
    # incarcam 2 exemple sql din /gold
    print(f"Loading few-shot examples...")
    few_shot_examples = load_few_shot_examples(eval_suite_dir, num_examples=2)
    print(f"Loaded {len(few_shot_examples)} gold SQL examples")
    
    # extractam id urile instatelor cu sql corect din /gold/sql
    gold_sql_dir = eval_suite_dir / "gold" / "sql"
    gold_instance_ids = {f.stem for f in gold_sql_dir.glob("local*.sql")}
    print(f"Found {len(gold_instance_ids)} instances with gold SQL")
    
    # Load data
    print(f"Loading data from {data_file}...")
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} total examples")
    
    # eliminam exemplele fara db sqlite sau fara sql corect
    # folosim doar local
    sqlite_data = []
    available_dbs = {f.stem for f in sqlite_files}
    skipped_count = 0
    
    for item in data:
        db_name = item['db']
        instance_id = item['instance_id']
        
        # Skip non-SQLite databases
        if db_name.startswith(('bigquery', 'snowflake', 'ga4', 'ga360', 'firebase')):
            skipped_count += 1
            continue
        
        # Skip if no gold SQL
        if instance_id not in gold_instance_ids:
            skipped_count += 1
            continue
        
        if db_name in available_dbs:
            sqlite_data.append(item)
    
    print(f"Skipped {skipped_count} examples (non-SQLite or no gold SQL)")
    print(f"Filtered {len(sqlite_data)} SQLite examples with gold SQL")
    
    if max_questions:
        sqlite_data = random.sample(sqlite_data, min(max_questions, len(sqlite_data)))
        print(f"Randomly selected {len(sqlite_data)} examples for testing")
    
    if not sqlite_data:
        print("No SQLite examples with gold SQL available!")
        return None
    
    predictions = []
    llm_fallback = None
    
    # structura pe care o sa rulam fiecare item
    for i, item in enumerate(sqlite_data):
        instance_id = item['instance_id']
        db_id = item['db']
        question = item['question']
        
        print(f"\n[{i+1}/{len(sqlite_data)}] {instance_id} | DB: {db_id}")
        
        db_file = db_dir / f"{db_id}.sqlite"
        
        if not db_file.exists():
            print(f"   Database not found")
            predictions.append({"instance_id": instance_id, "SQL": ""})
            continue
        
        try:
            import sqlite3
            
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_info = []
            for table in tables:
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
                create_stmt = cursor.fetchone()
                if create_stmt:
                    schema_info.append(create_stmt[0])
            
            conn.close()
            
            schema_text = "\n\n".join(schema_info)
            print(f"   {len(tables)} tables: {', '.join(tables[:4])}{'...' if len(tables) > 4 else ''}")
            
            few_shot_text = ""
            if few_shot_examples:
                few_shot_text = "\n\nExample SQL queries:\n\n"
                for ex in few_shot_examples:
                    few_shot_text += f"{ex['sql'][:200]}...\n\n"
            
            # fortam llm ul sa foloseasca doar tabelele din db
            # ofera tipuri de coloane
            # arata 2 exemple sql
            prompt_text = f"""You are an expert SQL query generator specializing in complex analytical queries.

CRITICAL INSTRUCTIONS:
1. Use ONLY these exact table names: {', '.join(tables)}
2. Write standard SQL compatible with SQLite
3. Return ONLY the complete SQL query, nothing else

DATABASE SCHEMA:
{schema_text}
{few_shot_text}

QUESTION: {question}

Generate a complete, executable SQL query:"""
            
            # Try primary LLM, fallback to Gemini
            generated_sql = None
            
            try:
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
                    print(f"   ⚠️  Rate limited - using Gemini fallback")
                    
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
            
            # curatam SQL generat
            generated_sql = re.sub(r'^```[\w]*', '', generated_sql, flags=re.MULTILINE)
            generated_sql = re.sub(r'```$', '', generated_sql, flags=re.MULTILINE)
            generated_sql = re.sub(r'^(SQL|Query):\s*', '', generated_sql, flags=re.IGNORECASE | re.MULTILINE)
            generated_sql = '\n'.join(line for line in generated_sql.split('\n') if line.strip())
            generated_sql = generated_sql.strip()
            
            try:
                test_conn = sqlite3.connect(str(db_file))
                test_cursor = test_conn.cursor()
                test_cursor.execute(f"EXPLAIN QUERY PLAN {generated_sql}")
                test_conn.close()
                print(f"   SQL valid")
            except Exception as e:
                print(f"   Syntax check failed: {str(e)[:50]}")
            
            print(f"   SQL: {generated_sql[:80]}...")
            
            predictions.append({
                "instance_id": instance_id,
                "SQL": generated_sql
            })
            
        except Exception as e:
            print(f"   Error: {str(e)[:80]}")
            predictions.append({
                "instance_id": instance_id,
                "SQL": ""
            })
    # salvam predictiile in formatul necesar evaluarii spider2-lite
    pred_file = Path("spider2_lite_predictions.jsonl")
    pred_dir = Path("spider2_lite_predictions")
    
    with open(pred_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    print(f"\nPredictions saved: {pred_file}")
    
    pred_dir.mkdir(exist_ok=True)
    
    for pred in predictions:
        instance_id = pred['instance_id']
        sql_content = pred['SQL']
        sql_file = pred_dir / f"{instance_id}.sql"
        with open(sql_file, 'w', encoding='utf-8') as f:
            f.write(sql_content)
    
    print(f"SQL files saved: {pred_dir}/ ({len(predictions)} files)")
    
    print("\n" + "="*60)
    print("Running official Spider2-Lite evaluation...")
    print("="*60)
    
    # rulam scriptul de evaluare oficial din spider2-lite
    import subprocess
    import sys
    
    cmd = [
        sys.executable,
        str(eval_script.resolve()),
        '--mode', 'sql',
        '--result_dir', str(pred_dir.absolute())
    ]
    
    print(f"Use: {' '.join(cmd)}\n")
    print(f"Evaluation in progress...")
    print(f"Results will be saved to: {pred_dir}/")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(eval_script.parent.resolve()),
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print(f"Stderr:\n{result.stderr}")
        
        if result.returncode != 0:
            print(f"\n  Evaluation script returned code {result.returncode}")
        
        # Parse score from output
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
        print(f"Evaluation timeout (5 minutes)")
        print(f"\n💡 Run manually:")
        print(f"   cd {eval_script.parent}")
        print(f"   python evaluate.py --mode sql --result_dir {pred_dir.absolute()}")
        
        return {
            'predictions': predictions,
            'predictions_file': str(pred_file),
            'predictions_dir': str(pred_dir),
            'timeout': True
        }
    except Exception as e:
        print(f"Evaluation error: {str(e)[:100]}")
        
        return {
            'predictions': predictions,
            'predictions_file': str(pred_file),
            'predictions_dir': str(pred_dir),
            'error': str(e)
        }

def load_few_shot_examples(eval_suite_dir: Path, num_examples: int = 3):
    gold_sql_dir = eval_suite_dir / "gold" / "sql"
    
    few_shot_examples = []
    
    # Select gold SQL examples
    gold_files = list(gold_sql_dir.glob("local*.sql"))[:num_examples]
    
    for sql_file in gold_files:
        instance_id = sql_file.stem
        sql_content = sql_file.read_text()
        
        few_shot_examples.append({
            'instance_id': instance_id,
            'sql': sql_content
        })
    
    return few_shot_examples

# ============================================================================
# 12. LANGGRAPH STATE MACHINE
# ============================================================================

# state graph directioant cu 5 noduri de procesare
# (detectie ambiguitate, cot, generare sql, validare sql, generare raspuns)
# cream graful de procesare cu clasa state care va trece intre noduri si fiecare nod o modifica

def create_text2sql_graph(use_cot: bool = True):
    graph_builder = StateGraph(State)
    
    # Add nodes in sequence
    graph_builder.add_node("detect_ambiguity", detect_ambiguity)
    
    if use_cot:
        graph_builder.add_node("chain_of_thought", generate_chain_of_thought)
    
    graph_builder.add_node("write_query", write_query)
    graph_builder.add_node("execute_query", validate_and_execute_query)
    graph_builder.add_node("generate_answer", generate_answer)
    
    graph_builder.add_edge(START, "detect_ambiguity")
    graph_builder.add_edge("detect_ambiguity", "chain_of_thought" if use_cot else "write_query")
    
    if use_cot:
        graph_builder.add_edge("chain_of_thought", "write_query")
    
    graph_builder.add_edge("write_query", "execute_query")
    graph_builder.add_edge("execute_query", "generate_answer")
    
    return graph_builder.compile()

# executam pipeline ul complet
def run_pipeline(question: str, use_cot: bool = True):
    graph = create_text2sql_graph(use_cot=use_cot)
    result = graph.invoke({"question": question})
    return result

# ============================================================================
# 12. CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="text2sql cli interface")
    parser.add_argument("--question", type=str, help="natural language question")
    parser.add_argument("--compare", action="store_true", help="compare different LLMs")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--evaluate", type=str, help="path to Spider JSON file")
    parser.add_argument("--spider2-lite", type=str, help="path to Spider2-Lite directory")
    parser.add_argument("--max-questions", type=int, default=None, help="maximum questions")
    parser.add_argument("--llm", type=str, default="groq", 
                       choices=["groq", "ollama", "ollama3.2", "ollama-qwen", "gemini", "cohere"],
                       help="default: groq")
    parser.add_argument("--no-cot", action="store_true", help="disable chain-of-thought")
    
    args = parser.parse_args()
    
    if not args.compare:
        print(f"LLM: {args.llm.upper()}")
        print(f"Chain-of-Thought: {'Disabled' if args.no_cot else 'Enabled'}")
    
    if args.spider2_lite:
        print("Running Spider2-Lite evaluation...")
        result = evaluate_on_spider2_lite(
            args.spider2_lite, 
            args.max_questions,
            primary_llm=args.llm
        )
        
        if result:
            print("\n" + "="*60)
            print("EVALUATION COMPLETE")
            print("="*60)
            print(f"Predictions saved to: {result.get('predictions_file')}")
            print(f"Total predictions: {len(result.get('predictions', []))}")
            
            if 'final_score' in result:
                print(f"\nFINAL SCORE: {result['final_score']:.2%}")
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
            question = input("\n\nEnter your question (or 'exit'): ")
            if question.lower() == 'exit':
                break
            
            result = run_pipeline(question, use_cot=not args.no_cot)
            print(f"\nAnswer: {result.get('answer', 'No answer generated')}")
    
    elif args.compare:
        question = args.question or input("Question for comparison: ")
        results = compare_llms(question)
        
        print("\n" + "="*60)
        print("LLM COMPARISON RESULTS")
        print("="*60)
        for model, result in results.items():
            print(f"\n{model.upper()}:")
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Query: {result.get('query', 'N/A')[:80]}...")
                print(f"  Answer: {result.get('answer', 'N/A')[:100]}...")
    
    else:
        question = args.question or input("\nQuestion: ")
        result = run_pipeline(question, use_cot=not args.no_cot)
        print(f"\nAnswer: {result.get('answer', 'No answer generated')}")

if __name__ == "__main__":
    main()