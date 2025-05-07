from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
import json
import csv
from langchain_ollama.llms import OllamaLLM
import re
from collections import Counter
import random

app = Flask(__name__)

# Set base directory for paths
BASE_DIR = "/Users/jagruthimekala/Library/Mobile Documents/com~apple~CloudDocs/rag"
PDF_PATH = os.path.join(BASE_DIR, "10th Maths - NCERT", "maths2.pdf")
CSV_PATH = os.path.join(BASE_DIR, "extracted_text.csv")

# Cache for extracted text to avoid re-processing
text_cache = None
topic_cache = {}
pattern_cache = {}

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, split into chunks, and save to CSV."""
    global text_cache
    
    # Return cached text if available
    if text_cache is not None:
        return text_cache
    
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
        
        # Cache the results
        text_cache = chunks
        return chunks
    
    except Exception as e:
        print(f"Error: Could not extract text from PDF: {str(e)}")
        return f"Error: Could not extract text from PDF: {str(e)}"

def load_chunks_from_csv():
    """Load chunks from CSV if PDF has already been processed."""
    global text_cache
    
    # Return cached text if available
    if text_cache is not None:
        return text_cache
    
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
        
        # Cache the results
        text_cache = chunks
        return chunks
    except Exception as e:
        print(f"Error loading from CSV: {str(e)}")
        return None

def extract_key_terms(chunks):
    """Extract key mathematical terms from chunks using pattern recognition."""
    if not chunks:
        return []
    
    # Common mathematical terms and patterns to look for
    math_patterns = [
        r'(?i)\b(equation|formula|theorem|law|principle|proof|function|algorithm)\b',
        r'(?i)\b(algebra|calculus|geometry|statistics|probability|trigonometry)\b',
        r'(?i)\b(integral|derivative|matrix|vector|scalar|graph|variable)\b',
        r'\b[A-Za-z]+\(.*?\)',  # Function notation like f(x)
        r'(?i)\b(sin|cos|tan|log|ln|exp)\b',  # Common math functions
        r'[\w\s]+ = [\w\s+\-*/^()]+',  # Equations with equals sign
    ]
    
    # Extract all terms matching patterns
    all_terms = []
    for chunk in chunks:
        content = chunk["content"]
        for pattern in math_patterns:
            matches = re.findall(pattern, content)
            if matches:
                all_terms.extend(matches)
    
    # Count term frequency
    term_counter = Counter(all_terms)
    
    # Return most common terms
    return [term for term, count in term_counter.most_common(20)]

def find_relevant_chunks(topic, chunks, num=5):
    """Find chunks relevant to a topic without using vector embeddings."""
    global pattern_cache
    
    # Create a pattern for the topic if not in cache
    if topic not in pattern_cache:
        # Convert topic to regex pattern with variations
        words = topic.lower().split()
        pattern_parts = []
        for word in words:
            if len(word) > 3:  # Only use significant words
                pattern_parts.append(r'\b' + word + r'\w*\b')  # Match word and variations
        
        if pattern_parts:
            pattern = '|'.join(pattern_parts)
            pattern_cache[topic] = re.compile(pattern, re.IGNORECASE)
        else:
            # Fallback for short topics
            pattern_cache[topic] = re.compile(re.escape(topic), re.IGNORECASE)
    
    pattern = pattern_cache[topic]
    
    # Score chunks based on pattern matches
    scored_chunks = []
    for chunk in chunks:
        content = chunk["content"]
        # Count matches
        matches = len(re.findall(pattern, content))
        if matches > 0:
            # Score is based on match count and content length
            score = matches * (1000 / (len(content) + 200))  # Normalize by length, favor shorter content
            scored_chunks.append((score, chunk))
    
    # Sort by score and take top chunks
    scored_chunks.sort(reverse=True)
    return [chunk for _, chunk in scored_chunks[:num]]

def generate_mcq_with_vibe(topic, context_chunks, num_questions=5):
    """Generate quiz questions using vibes approach with Ollama."""
    model = OllamaLLM(model="llama3.2")
    
    # Extract specific context from chunks
    context = "\n\n".join([chunk["content"] for chunk in context_chunks])
    
    template = f"""
    You are an expert senior maths tutor specialized in creating maths questions with explanations step by step.
    
    Here is the educational content to base your questions on:
    
    {context}
    
    TOPIC: {topic}
    
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
    
    try:
        result = model.invoke(template)
        
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
                        print(valid_questions)
                        return valid_questions[:num_questions]  # Return only requested number
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Attempted to parse: {json_str[:100]}...")
        
        return {"error": "Failed to generate proper JSON response from the model."}
    
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return {"error": f"Failed to generate questions: {str(e)}"}

def generate_topic_based_questions(topic, chunks, num_questions=5):
    """Generate questions based on a specific topic."""
    # First find relevant chunks for the topic
    relevant_chunks = find_relevant_chunks(topic, chunks, num=5)
    
    if not relevant_chunks:
        return {"error": "No relevant content found for this topic."}
    
    # Generate questions based on the chunks
    return generate_mcq_with_vibe(topic, relevant_chunks, num_questions)

@app.route('/')
def index():
    """Load the quiz webpage."""
    return render_template('quiz.html')

@app.route('/api/quiz', methods=['GET'])
def get_quiz():
    """Generate and return quiz questions."""
    # Load chunks from CSV or extract from PDF
    chunks = load_chunks_from_csv()
    if not chunks:
        chunks = extract_text_from_pdf(PDF_PATH)
        if isinstance(chunks, str) and "Error" in chunks:
            return jsonify({"error": chunks}), 500
    
    num_questions = request.args.get('num', 5, type=int)
    topic = request.args.get('topic', '')
    
    print(f"Generating quiz with topic: '{topic}', num questions: {num_questions}")
    
    # If topic is provided, generate topic-specific questions
    if topic:
        if topic in topic_cache:
            print(f"Using cached questions for topic: {topic}")
            return jsonify(topic_cache[topic])
            
        quiz_questions = generate_topic_based_questions(topic, chunks, num_questions)
    else:
        # Otherwise, create a generic topic to retrieve a good sample of content
        generic_topic = "key concepts and important information"
        print(f"Using generic topic: '{generic_topic}'")
        quiz_questions = generate_topic_based_questions(generic_topic, chunks, num_questions)
    
    # Check if we got an error instead of questions
    if isinstance(quiz_questions, dict) and "error" in quiz_questions:
        print(f"Error generating quiz: {quiz_questions['error']}")
        return jsonify(quiz_questions), 500
    
    # Cache the results for this topic
    if topic:
        topic_cache[topic] = quiz_questions
        
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
    """Extract potential topics from the document."""
    # Load chunks
    chunks = load_chunks_from_csv()
    if not chunks:
        chunks = extract_text_from_pdf(PDF_PATH)
        if isinstance(chunks, str) and "Error" in chunks:
            return jsonify({"error": chunks}), 500
    
    # Extract key terms using pattern matching
    key_terms = extract_key_terms(chunks)
    
    # If we found terms, return them
    if key_terms:
        # Group related terms and clean up
        grouped_terms = []
        seen_patterns = set()
        
        for term in key_terms:
            # Skip terms that are too short or already seen
            if len(term) < 4 or term.lower() in seen_patterns:
                continue
                
            seen_patterns.add(term.lower())
            if len(term) > 3:  # Only use significant terms
                grouped_terms.append(term.capitalize())
        
        # Return top topics (limit to 8)
        return jsonify(grouped_terms[:8])
    
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
        
        if "operational" in test_result.lower():
            # Check if CSV exists
            csv_exists = os.path.exists(CSV_PATH)
            if csv_exists:
                df = pd.read_csv(CSV_PATH)
                csv_status = f"CSV storage ready with {len(df)} rows"
            else:
                csv_status = "Will create new CSV file"
            
            return jsonify({
                "status": "success",
                "message": "All systems operational (Vibe mode)",
                "mode": "Using fast pattern matching",
                "csv_status": csv_status
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"System check failed: {str(e)}",
            "hint": "Make sure Ollama is running with llama3.2 model"
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
    global text_cache, topic_cache, pattern_cache
    
    # Clear caches
    text_cache = None
    topic_cache = {}
    pattern_cache = {}
    
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