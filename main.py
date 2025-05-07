from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
import json
import csv
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Set base directory for paths
BASE_DIR = "/Users/jagruthimekala/Library/Mobile Documents/com~apple~CloudDocs/rag"
PDF_PATH = os.path.join(BASE_DIR, "10th Maths - NCERT", "maths2.pdf")
DB_LOCATION = os.path.join(BASE_DIR, "chroma_langchain_db")
CSV_PATH = os.path.join(BASE_DIR, "extracted_text.csv")



def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, split into chunks, and save to CSV."""
    if not os.path.exists(pdf_path):
        return "Error: PDF file not found."
    
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        
        # Write to CSV file
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['source', 'page', 'chunk', 'content']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()
                # Split text into smaller paragraphs
                paragraphs = [p for p in text.split('\n\n') if p.strip()]
                
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        # Create a chunk dictionary
                        chunk = {
                            "content": paragraph,
                            "metadata": {
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1,
                                "chunk": i + 1
                            }
                        }
                        chunks.append(chunk)
                        
                        # Write to CSV
                        writer.writerow({
                            'source': os.path.basename(pdf_path),
                            'page': page_num + 1,
                            'chunk': i + 1,
                            'content': paragraph
                        })
        
        print(f"Successfully extracted text from PDF and saved to {CSV_PATH}")
        
        # Verify that data was written to CSV
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            print(f"CSV file created with {len(df)} rows")
        else:
            print("CSV file was not created")
        
        return chunks
    except Exception as e:
        print(f"Error: Could not extract text from PDF: {str(e)}")
        return f"Error: Could not extract text from PDF: {str(e)}"

def load_chunks_from_csv():
    """Load chunks from CSV if PDF has already been processed."""
    if not os.path.exists(CSV_PATH):
        return None
    
    try:
        chunks = []
        df = pd.read_csv(CSV_PATH)
        
        for _, row in df.iterrows():
            chunk = {
                "content": row["content"],
                "metadata": {
                    "source": row["source"],
                    "page": row["page"],
                    "chunk": row["chunk"]
                }
            }
            chunks.append(chunk)
        
        return chunks
    except Exception as e:
        print(f"Error loading from CSV: {str(e)}")
        return None

def setup_vector_store():
    """Create and populate the vector store with document chunks."""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # First try to load chunks from CSV if it exists
    chunks = load_chunks_from_csv()
    
    # If CSV doesn't exist or is empty, extract from PDF
    if not chunks:
        chunks = extract_text_from_pdf(PDF_PATH)
        if isinstance(chunks, str) and "Error" in chunks:
            print(chunks)
            return None
    
    # Create directory for vector store if it doesn't exist
    os.makedirs(DB_LOCATION, exist_ok=True)
    
    # Check if we need to add documents
    add_documents = not os.path.exists(os.path.join(DB_LOCATION, "chroma.sqlite3"))
    
    # Initialize vector store
    vector_store = Chroma(
        collection_name="pdf_content",
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )
    
    # Add documents if needed
    if add_documents and chunks:
        try:
            documents = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                document = Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"],
                )
                ids.append(str(i))
                documents.append(document)
            
            # Only add if there are documents to add
            if documents:
                print(f"Adding {len(documents)} documents to vector store")
                vector_store.add_documents(documents=documents, ids=ids)
                vector_store.persist()
                print("Documents added to vector store successfully")
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
    
    return vector_store

def retrieve_relevant_content(question, vector_store, k=5):
    """Retrieve relevant content based on the question."""
    if not vector_store:
        return "Error: Vector store not initialized."
    
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        
        # Format the retrieved documents
        context = []
        for i, doc in enumerate(docs):
            context.append(f"Document {i+1} (Page {doc.metadata['page']}):\n{doc.page_content}")
        
        return "\n\n".join(context)
    except Exception as e:
        print(f"Error retrieving content: {str(e)}")
        return f"Error retrieving content: {str(e)}"

def generate_mcq_with_rag(question_prompt, context, num_questions=5):
    """Generate quiz questions using RAG approach with Ollama."""
    model = OllamaLLM(model="llama3.2")
    
    template = """
    You are an expert quiz creator specialized in creating educational multiple-choice questions.
    
    Here is the educational content to base your questions on:
    
    {context}
    
    INSTRUCTIONS:
    1. Create exactly {num_questions} quiz questions based ONLY on the content provided
    2. Each question must have exactly 4 options labeled A, B, C, and D
    3. Specify which option is correct
    4. Provide a brief explanation for the correct answer
    5. Make sure the questions test understanding, not just memorization
    
    FORMAT YOUR RESPONSE AS VALID JSON:
    [
      {{
        "question": "Question text here?",
        "options": [
          "A. First option",
          "B. Second option",
          "C. Third option",
          "D. Fourth option"
        ],
        "answer": "B. Second option",
        "explanation": "Explanation for why B is correct"
      }},
      ... (more questions)
    ]
    
    YOUR RESPONSE MUST BE VALID JSON ONLY, NO OTHER TEXT.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    try:
        result = model.invoke(prompt.format(
            context=context, 
            num_questions=num_questions
        ))
        
        # Extract JSON from the response
        json_start = result.find('[')
        json_end = result.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = result[json_start:json_end]
            # Fix common JSON formatting issues
            json_str = json_str.replace('\n', ' ')
            
            try:
                # Parse the JSON
                quiz_data = json.loads(json_str)
                
                # Validate the structure
                if isinstance(quiz_data, list) and len(quiz_data) > 0:
                    valid_questions = []
                    for q in quiz_data:
                        if all(k in q for k in ["question", "options", "answer", "explanation"]):
                            if len(q["options"]) == 4:  # Ensure 4 options
                                valid_questions.append(q)
                    
                    if valid_questions:
                        return valid_questions[:num_questions]  # Return only requested number
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Attempted to parse: {json_str[:100]}...")
        
        return {"error": "Failed to generate proper JSON response from the model."}
    
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return {"error": f"Failed to generate questions: {str(e)}"}

def generate_topic_based_questions(topic, vector_store, num_questions=5):
    """Generate questions based on a specific topic."""
    # First retrieve relevant content for the topic
    context = retrieve_relevant_content(topic, vector_store)
    
    if isinstance(context, str) and context.startswith("Error"):
        return {"error": context}
    
    if not context:
        return {"error": "No relevant content found for this topic."}
    
    # Generate questions based on the retrieved content
    return generate_mcq_with_rag(topic, context, num_questions)

@app.route('/')
def index():
    """Load the quiz webpage."""
    return render_template('quiz1.2.html')

@app.route('/api/quiz', methods=['GET'])
def get_quiz():
    """Generate and return quiz questions."""
    # Initialize or retrieve the vector store
    vector_store = setup_vector_store()
    
    if not vector_store:
        return jsonify({"error": "Failed to initialize vector store."}), 500
    
    num_questions = request.args.get('num', 5, type=int)
    topic = request.args.get('topic', '')
    
    print(f"Generating quiz with topic: '{topic}', num questions: {num_questions}")
    
    # If topic is provided, generate topic-specific questions
    if topic:
        # Retrieve content first to debug
        context = retrieve_relevant_content(topic, vector_store)
        print(f"Retrieved context length: {len(context) if isinstance(context, str) else 'Error'}")
        if isinstance(context, str) and context.startswith("Error"):
            return jsonify({"error": context}), 500
        if not context or context.strip() == "":
            return jsonify({"error": "No relevant content found for this topic."}), 500
            
        quiz_questions = generate_topic_based_questions(topic, vector_store, num_questions)
    else:
        # Otherwise, create a generic topic to retrieve a good sample of content
        generic_topic = "key concepts and important information"
        print(f"Using generic topic: '{generic_topic}'")
        
        # Retrieve content first to debug
        context = retrieve_relevant_content(generic_topic, vector_store)
        print(f"Retrieved context length: {len(context) if isinstance(context, str) else 'Error'}")
        if isinstance(context, str) and context.startswith("Error"):
            return jsonify({"error": context}), 500
        if not context or context.strip() == "":
            return jsonify({"error": "No relevant content found using generic topic."}), 500
            
        quiz_questions = generate_topic_based_questions(generic_topic, vector_store, num_questions)
    
    # Check if we got an error instead of questions
    if isinstance(quiz_questions, dict) and "error" in quiz_questions:
        print(f"Error generating quiz: {quiz_questions['error']}")
        return jsonify(quiz_questions), 500
        
    print(f"Successfully generated {len(quiz_questions)} questions")
    return jsonify(quiz_questions)

@app.route('/api/submit', methods=['POST'])
def submit_quiz():
    """Evaluate submitted quiz answers and return the score with explanations."""
    if request.content_type != 'application/json':
        return jsonify({"error": "Unsupported Media Type"}), 415
    
    data = request.json
    questions = data.get("questions", [])
    user_answers = data.get("answers", {})
    
    score = 0
    feedback = []
    
    for question_text, user_answer in user_answers.items():
        # Find the matching question
        matching_question = None
        for q in questions:
            if q["question"] == question_text:
                matching_question = q
                break
        
        if matching_question:
            correct_answer = matching_question.get("answer", "")
            is_correct = user_answer == correct_answer
            
            if is_correct:
                score += 1
            
            feedback.append({
                "question": question_text,
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "explanation": matching_question.get("explanation", "No explanation available."),
                "correct": is_correct
            })
    
    return jsonify({
        "score": score, 
        "total": len(feedback), 
        "feedback": feedback
    })

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Extract potential topics from the document for the user to choose from."""
    # Initialize or retrieve the vector store
    vector_store = setup_vector_store()
    
    if not vector_store:
        return jsonify({"error": "Failed to initialize vector store."}), 500
    
    # Get a sample of documents
    try:
        docs = vector_store.similarity_search("main topics important concepts", k=10)
        
        # Use LLM to extract topics
        model = OllamaLLM(model="llama3.2")
        
        content = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = """
        Based on the following content from an educational document, list the 5-8 main topics covered.
        Format your response as a JSON array of strings. ONLY return the JSON array, nothing else.
        
        Document content:
        {content}
        
        Return format: ["Topic 1", "Topic 2", "Topic 3", ...]
        """
        
        try:
            result = model.invoke(prompt.format(content=content))
            
            # Try to extract JSON
            json_start = result.find('[')
            json_end = result.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                topics = json.loads(json_str)
                return jsonify(topics)
        except Exception as e:
            print(f"Error extracting topics: {str(e)}")
            
    except Exception as e:
        print(f"Error in similarity search: {str(e)}")
    
    # Fallback topics if extraction fails
    default_topics = ["Basic Concepts", "Key Formulas", "Problem Solving", "Applications"]
    return jsonify(default_topics)

@app.route('/api/debug/status', methods=['GET'])
def debug_status():
    """Check if all systems are operational."""
    try:
        # Test if Ollama is working
        model = OllamaLLM(model="llama3.2")
        test_result = model.invoke("Say 'System is operational'")
        
        # Test if embeddings are working
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        embed_result = embeddings.embed_query("test")
        
        if len(embed_result) > 0 and "operational" in test_result.lower():
            # Check if vector DB exists
            db_exists = os.path.exists(os.path.join(DB_LOCATION, "chroma.sqlite3"))
            mode = "With existing DB" if db_exists else "Will create new DB"
            
            # Check if CSV exists
            csv_exists = os.path.exists(CSV_PATH)
            if csv_exists:
                df = pd.read_csv(CSV_PATH)
                csv_status = f"CSV storage ready with {len(df)} rows"
            else:
                csv_status = "Will create new CSV file"
            
            return jsonify({
                "status": "success",
                "message": "All systems operational",
                "mode": mode,
                "csv_status": csv_status
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"System check failed: {str(e)}",
            "hint": "Make sure Ollama is running with llama3.2 and mxbai-embed-large models"
        })

@app.route('/api/extracted-text', methods=['GET'])
def get_extracted_text():
    """Return the extracted text stored in the CSV file."""
    if not os.path.exists(CSV_PATH):
        # Try to extract text from PDF if CSV doesn't exist
        result = extract_text_from_pdf(PDF_PATH)
        if isinstance(result, str) and "Error" in result:
            return jsonify({"error": result}), 404
    
    try:
        if os.path.exists(CSV_PATH):
            data = pd.read_csv(CSV_PATH)
            return jsonify(data.to_dict(orient='records'))
        else:
            return jsonify({"error": "Failed to create CSV file."}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to read extracted text: {str(e)}"}), 500

@app.route('/api/force-extract', methods=['GET'])
def force_extract():
    """Force re-extraction of text from PDF."""
    try:
        chunks = extract_text_from_pdf(PDF_PATH)
        if isinstance(chunks, str) and "Error" in chunks:
            return jsonify({"error": chunks}), 500
        
        return jsonify({
            "status": "success",
            "message": f"Successfully extracted {len(chunks)} chunks from PDF",
            "csv_path": CSV_PATH
        })
    except Exception as e:
        return jsonify({"error": f"Failed to extract text: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)