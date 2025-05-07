from flask import Flask, render_template, request, jsonify, Response
import os
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
import json
import csv
from langchain_ollama.llms import OllamaLLM
import re
from collections import Counter
import threading
import queue
import time

app = Flask(__name__)






# Set base directory for paths
BASE_DIR = "/Users/jagruthimekala/Library/Mobile Documents/com~apple~CloudDocs/rag"
PDF_PATH = os.path.join(BASE_DIR, "10th Maths - NCERT", "maths2.pdf")
CSV_PATH = os.path.join(BASE_DIR, "extracted_text.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Enhanced caching
text_cache = None
chunk_index_cache = {}  # Maps topics to relevant chunk indices
topic_cache = {}
pattern_cache = {}
key_terms_cache = None

# Preprocessing & indexing constants
MIN_CHUNK_LENGTH = 50
MAX_CHUNKS_FOR_CONTEXT = 5
CHUNK_RELEVANCE_THRESHOLD = 0.1

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, split into optimized chunks, and save to CSV."""
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
            fieldnames = ['source', 'page', 'chunk', 'content', 'keywords']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()
                
                # Extract semantic paragraphs more intelligently
                paragraphs = []
                current_paragraph = []
                
                # Split by newlines but preserve semantic paragraphs
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line and current_paragraph:
                        # End of paragraph
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                    elif line:
                        current_paragraph.append(line)
                
                # Don't forget the last paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                
                # Filter out very short chunks and extract keywords for each chunk
                for i, paragraph in enumerate(paragraphs):
                    if len(paragraph.strip()) < MIN_CHUNK_LENGTH:
                        continue
                        
                    # Simple keyword extraction
                    keywords = extract_keywords(paragraph)
                    keywords_str = ','.join(keywords)
                    
                    # Create a chunk dictionary
                    chunk = {
                        "content": paragraph,
                        "metadata": {
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "chunk": i + 1,
                            "keywords": keywords
                        }
                    }
                    chunks.append(chunk)
                    
                    # Write to CSV
                    writer.writerow({
                        'source': os.path.basename(pdf_path),
                        'page': page_num + 1,
                        'chunk': i + 1,
                        'content': paragraph,
                        'keywords': keywords_str
                    })
        
        print(f"Successfully extracted text from PDF and saved to {CSV_PATH}")
        
        # Cache the results
        text_cache = chunks
        # Create index for faster lookups
        build_chunk_index(chunks)
        return chunks
    
    except Exception as e:
        print(f"Error: Could not extract text from PDF: {str(e)}")
        return f"Error: Could not extract text from PDF: {str(e)}"

def build_chunk_index(chunks):
    """Build an index mapping keywords to chunk indices for faster lookups."""
    global chunk_index_cache
    
    chunk_index_cache = {}
    for i, chunk in enumerate(chunks):
        for keyword in chunk["metadata"]["keywords"]:
            if keyword not in chunk_index_cache:
                chunk_index_cache[keyword] = []
            chunk_index_cache[keyword].append(i)

def extract_keywords(text):
    """Extract keywords from text for indexing."""
    # Common mathematical terms to identify
    math_terms = [
        "equation", "formula", "theorem", "law", "principle", "proof", "function", 
        "algorithm", "algebra", "calculus", "geometry", "statistics", "probability", 
        "trigonometry", "integral", "derivative", "matrix", "vector", "scalar", 
        "graph", "variable", "sin", "cos", "tan", "log", "ln", "exp"
    ]
    
    # Find all words
    words = re.findall(r'\b[a-z][a-z0-9_]{2,}\b', text.lower())
    
    # Prioritize mathematical terms
    keywords = [w for w in words if w in math_terms]
    
    # Add additional significant words (non-stopwords)
    stopwords = {"the", "and", "is", "in", "to", "of", "that", "it", "with", "as", "for", "on", "was", "be", "by"}
    keywords.extend([w for w in words if w not in stopwords and w not in keywords])
    
    # Limit to top keywords
    return list(set(keywords))[:10]  # Use set to remove duplicates

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
            # Handle keywords column
            keywords = []
            if 'keywords' in row and isinstance(row["keywords"], str):
                keywords = row["keywords"].split(',')
                
            chunk = {
                "content": row["content"],
                "metadata": {
                    "source": row["source"],
                    "page": row["page"],
                    "chunk": row["chunk"],
                    "keywords": keywords
                }
            }
            chunks.append(chunk)
        
        # Cache the results
        text_cache = chunks
        # Build index for faster lookups
        build_chunk_index(chunks)
        return chunks
    except Exception as e:
        print(f"Error loading from CSV: {str(e)}")
        return None

def extract_key_terms(chunks):
    """Extract key mathematical terms from chunks using pattern recognition."""
    global key_terms_cache
    
    # Return cached terms if available
    if key_terms_cache is not None:
        return key_terms_cache
        
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
    
    # Get most common terms
    result = [term for term, count in term_counter.most_common(20)]
    key_terms_cache = result
    return result

def find_relevant_chunks(topic, chunks, num=MAX_CHUNKS_FOR_CONTEXT):
    """Find chunks relevant to a topic using optimized matching."""
    global pattern_cache, chunk_index_cache
    
    # Check if we have cached chunk indices for this topic
    topic_key = topic.lower().strip()
    if topic_key in chunk_index_cache:
        indices = chunk_index_cache[topic_key]
        return [chunks[i] for i in indices[:num]]
    
    # Convert topic to search terms
    search_terms = set()
    words = topic.lower().split()
    
    # Add each significant word as a search term
    for word in words:
        if len(word) > 3:  # Only use significant words
            search_terms.add(word)
    
    # If we have an index, use it for faster lookups
    if chunk_index_cache:
        candidate_indices = set()
        for term in search_terms:
            if term in chunk_index_cache:
                candidate_indices.update(chunk_index_cache[term])
            
            # Try partial matches too
            for keyword in chunk_index_cache:
                if term in keyword or keyword in term:
                    candidate_indices.update(chunk_index_cache[keyword])
        
        # Filter to only candidate chunks
        candidate_chunks = [chunks[i] for i in candidate_indices]
    else:
        # Fallback to checking all chunks
        candidate_chunks = chunks
    
    # Create a pattern for the topic if not in cache
    if topic not in pattern_cache:
        # Convert topic to regex pattern with variations
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
    
    # Score chunks based on pattern matches - optimized scoring
    scored_chunks = []
    for chunk in candidate_chunks:
        content = chunk["content"]
        # Count matches
        matches = len(re.findall(pattern, content))
        
        # Score is based on match count, density, and content length
        if matches > 0:
            score = (matches / (len(content.split()) + 50)) * 1000
            
            # Bonus for keywords in metadata that match
            if "keywords" in chunk["metadata"]:
                for keyword in chunk["metadata"]["keywords"]:
                    if any(term in keyword for term in search_terms):
                        score += 100
                        
            if score >= CHUNK_RELEVANCE_THRESHOLD:
                scored_chunks.append((score, chunk))
    
    # Sort by score and take top chunks
    scored_chunks.sort(reverse=True)
    result = [chunk for _, chunk in scored_chunks[:num]]
    
    # Cache the indices for future use
    chunk_index_cache[topic_key] = [chunks.index(chunk) for _, chunk in scored_chunks[:num]]
    
    return result

# Global queue for streaming results
stream_queue = queue.Queue()

def generate_question_worker(topic, context_chunks, num_questions=5):
    """Worker thread for generating quiz questions."""
    try:
        results = generate_mcq_with_vibe(topic, context_chunks, num_questions)
        # Put result in queue
        stream_queue.put({"status": "complete", "data": results})
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        stream_queue.put({"status": "error", "message": str(e)})

def generate_mcq_with_vibe(topic, context_chunks, num_questions=5):
    """Generate quiz questions using optimized vibes approach with Ollama."""
    model = OllamaLLM(model="llama3.2")
    
    # Extract specific context from chunks
    context = "\n\n".join([chunk["content"] for chunk in context_chunks])
    
    # Create a more efficient prompt
    template = f"""
    Create {num_questions} mathematics quiz questions on "{topic}" based on this content:
    
    {context}
    
    For each question:
    - Include 4 options (A, B, C, D)
    - Indicate the correct answer
    - Provide a brief explanation
    
    Format as JSON array:
    [
      {{
        "question": "...",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "answer": "A. ...",
        "explanation": "..."
      }}
    ]
    
    Return ONLY valid JSON.
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
                
                # Validate and clean up the structure
                if isinstance(quiz_data, list) and len(quiz_data) > 0:
                    valid_questions = []
                    for q in quiz_data:
                        if all(k in q for k in ["question", "options", "answer", "explanation"]):
                            if len(q["options"]) == 4:  # Ensure 4 options
                                # Make sure options start with A, B, C, D
                                fixed_options = []
                                for i, opt in enumerate(q["options"]):
                                    if not opt.startswith(f"{chr(65+i)}. "):  # A, B, C, D
                                        fixed_options.append(f"{chr(65+i)}. {opt.lstrip('ABCD. ')}")
                                    else:
                                        fixed_options.append(opt)
                                
                                q["options"] = fixed_options
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

def generate_topic_based_questions(topic, chunks, num_questions=5):
    """Generate questions based on a specific topic with optimized context retrieval."""
    # First find relevant chunks for the topic
    relevant_chunks = find_relevant_chunks(topic, chunks, num=MAX_CHUNKS_FOR_CONTEXT)
    
    if not relevant_chunks:
        return {"error": "No relevant content found for this topic."}
    
    # Generate questions based on the chunks
    return generate_mcq_with_vibe(topic, relevant_chunks, num_questions)

@app.route('/')
def index():
    """Load the quiz webpage."""
    return render_template('quiz5.html')

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
    
    # Check cache first
    if topic and topic in topic_cache:
        print(f"Using cached questions for topic: {topic}")
        return jsonify(topic_cache[topic])
            
    # Launch question generation in a separate thread
    if topic:
        relevant_chunks = find_relevant_chunks(topic, chunks, num=MAX_CHUNKS_FOR_CONTEXT)
    else:
        # Use a generic topic to retrieve a good sample of content
        generic_topic = "key concepts and important information"
        print(f"Using generic topic: '{generic_topic}'")
        relevant_chunks = find_relevant_chunks(generic_topic, chunks, num=MAX_CHUNKS_FOR_CONTEXT)
    
    if not relevant_chunks:
        return jsonify({"error": "No relevant content found for this topic."}), 500
    
    # Generate questions
    actual_topic = topic if topic else "key concepts"
    quiz_questions = generate_mcq_with_vibe(actual_topic, relevant_chunks, num_questions)
    
    # Check if we got an error instead of questions
    if isinstance(quiz_questions, dict) and "error" in quiz_questions:
        print(f"Error generating quiz: {quiz_questions['error']}")
        return jsonify(quiz_questions), 500
    
    # Cache the results for this topic
    if topic:
        topic_cache[topic] = quiz_questions
        
    print(f"Successfully generated {len(quiz_questions)} questions")
    return jsonify(quiz_questions)

@app.route('/api/quiz-stream', methods=['GET'])
def stream_quiz():
    """Stream quiz questions as they're generated."""
    c=0
    
    # Load chunks from CSV or extract from PDF
    chunks = load_chunks_from_csv()
    if not chunks:
        chunks = extract_text_from_pdf(PDF_PATH)
        if isinstance(chunks, str) and "Error" in chunks:
            return jsonify({"error": chunks}), 500
    
    num_questions = request.args.get('num', 5, type=int)
    topic = request.args.get('topic', '')
    
    print(f"Streaming quiz with topic: '{topic}', num questions: {num_questions}")
    
    # Clear the queue
    while not stream_queue.empty():
        stream_queue.get()
    
    # Find relevant chunks
    if topic:
        relevant_chunks = find_relevant_chunks(topic, chunks, num=MAX_CHUNKS_FOR_CONTEXT)
    else:
        # Use a generic topic
        generic_topic = "key concepts and important information"
        relevant_chunks = find_relevant_chunks(generic_topic, chunks, num=MAX_CHUNKS_FOR_CONTEXT)
    
    if not relevant_chunks:
        return jsonify({"error": "No relevant content found for this topic."}), 500
    
    # Start the worker thread
    actual_topic = topic if topic else "key concepts"
    threading.Thread(
        target=generate_question_worker, 
        args=(actual_topic, relevant_chunks, num_questions)
    ).start()
    
    def generate():
        """Generator function for the streaming response."""
        while True:
            try:
                # Wait for new data with timeout
                item = stream_queue.get(timeout=0.5)
                print(item)
                yield f"data: {json.dumps(item)}\n\n"

                
                # If complete or error, stop streaming
                if item.get("status") in ["complete", "error"]:
                    break
            except queue.Empty:
                # Send a keepalive
                yield "data: {\"status\": \"working\"}\n\n"
        
    return Response(generate(), mimetype='text/event-stream')

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
            correct_letter = correct_answer.split('.')[0].strip() if '.' in correct_answer else correct_answer
            user_letter = user_answer.strip() if user_answer else ""
            
            is_correct = user_letter == correct_letter
            
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
    """Extract potential topics from the document with improved caching."""
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
            if isinstance(term, str):
                term_text = term
            else:
                # Handle case where term might be a tuple from regex
                term_text = term[0] if isinstance(term, tuple) else str(term)
                
            if len(term_text) < 4 or term_text.lower() in seen_patterns:
                continue
                
            seen_patterns.add(term_text.lower())
            if len(term_text) > 3:  # Only use significant terms
                grouped_terms.append(term_text.capitalize())
        
        # Return top topics (limit to 10)
        return jsonify(grouped_terms[:10])
    
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
            
            # Check cache status
            cache_status = {
                "text_cache": "Active" if text_cache is not None else "Empty",
                "topic_cache": f"{len(topic_cache)} topics" if topic_cache else "Empty",
                "pattern_cache": f"{len(pattern_cache)} patterns" if pattern_cache else "Empty",
                "chunk_index": f"{len(chunk_index_cache)} terms indexed" if chunk_index_cache else "Not built"
            }
            
            return jsonify({
                "status": "success",
                "message": "All systems operational (Enhanced Vibe mode)",
                "mode": "Using optimized pattern matching + keyword indexing",
                "csv_status": csv_status,
                "cache_status": cache_status
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
    global text_cache, topic_cache, pattern_cache, key_terms_cache, chunk_index_cache
    
    # Clear all caches
    text_cache = None
    topic_cache = {}
    pattern_cache = {}
    key_terms_cache = None
    chunk_index_cache = {}
    
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