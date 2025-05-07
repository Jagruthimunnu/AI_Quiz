import os
import sys
import fitz  # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import argparse
from typing import List, Dict, Any, Tuple
import ollama
import time
import random
import traceback
from flask import Flask, render_template, request, jsonify, send_from_directory

class PDFQuizGenerator:
    def __init__(self, model_name="llama3.2"):
        """Initialize the Quiz Generator with Ollama model"""
        self.model_name = model_name
        print(f"Initializing with model: {model_name}")
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            if end < text_length and end - start == chunk_size:
                # Find the last period or newline to make better chunks
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:  # Only use if it's a reasonable point
                    end = break_point + 1
            
            chunks.append(text[start:end])
            start = end - overlap if end < text_length else text_length
            
        return chunks
        
    def get_vibe_encoding(self, text: str) -> List[Dict[str, Any]]:
        """
        Create 'vibe encoding' of the text using key phrases and concepts
        This is a simplified alternative to embedding-based RAG
        """
        # Get key phrases from Ollama
        prompt = f"""
        Extract the key phrases, concepts, and terminology from the following text. 
        For each key phrase, assign an importance score (1-10) and identify related concepts.
        Format your response as a JSON array of objects with the following fields:
        - phrase: the key phrase or concept
        - importance: numeric score 1-10
        - related: array of related concepts
        Text to analyze:
        {text[:3000]}  # Limit text size for prompt
        """
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json"
            )
            
            content = response['message']['content']
            
            # Extract the JSON part from the response
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
            
            # Clean up any markdown or extra text
            content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            
            # Try to find a JSON array in the text
            match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
            if match:
                content = match.group(0)
                
            try:
                vibe_data = json.loads(content)
                if isinstance(vibe_data, list):
                    return vibe_data
                return []
            except json.JSONDecodeError:
                # If standard parsing fails, try a more lenient approach
                print("Initial JSON parsing failed, attempting to clean and reparse...")
                # Simple cleanup - replace single quotes with double quotes, etc.
                content = content.replace("'", "\"").replace("None", "null")
                try:
                    vibe_data = json.loads(content)
                    if isinstance(vibe_data, list):
                        return vibe_data
                    return []
                except:
                    print("Failed to parse JSON response, returning empty vibe encoding")
                    return []
                    
        except Exception as e:
            print(f"Error getting vibe encoding: {e}")
            return []
    
    def get_relevant_chunks(self, query: str, chunks: List[str], 
                           vibe_encodings: List[List[Dict[str, Any]]],
                           top_k: int = 3) -> List[str]:
        """Find most relevant chunks based on vibe similarity"""
        # Extract key terms from the query
        try:
            prompt = f"""
            Extract the 5-10 most important key terms and concepts from this question:
            {query}
            Format your response as a simple JSON array of strings, just the terms themselves.
            """
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json"
            )
            
            content = response['message']['content']
            
            # Extract JSON from response
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                content = match.group(0)
                
            try:
                query_terms = json.loads(content)
                if not isinstance(query_terms, list):
                    # Fall back to simple term extraction
                    query_terms = [term.strip().lower() for term in re.findall(r'[\w\s]{3,}', query.lower()) 
                                 if len(term.strip()) > 3]
            except json.JSONDecodeError:
                # If parsing fails, try to extract terms manually
                content = content.replace("'", "\"")
                try:
                    query_terms = json.loads(content)
                    if not isinstance(query_terms, list):
                        query_terms = [term.strip().lower() for term in re.findall(r'[\w\s]{3,}', query.lower()) 
                                     if len(term.strip()) > 3]
                except:
                    # Fall back to simple term extraction
                    query_terms = [term.strip().lower() for term in re.findall(r'[\w\s]{3,}', query.lower()) 
                                 if len(term.strip()) > 3]
            
            # Score chunks based on term overlap with vibe encodings
            scores = []
            for i, encoding in enumerate(vibe_encodings):
                score = 0
                for term in query_terms:
                    term_lower = term.lower()
                    for item in encoding:
                        phrase = str(item.get('phrase', '')).lower()
                        importance = item.get('importance', 5)
                        
                        # Score direct matches
                        if term_lower in phrase or phrase in term_lower:
                            score += importance
                        
                        # Score related concept matches
                        related_list = item.get('related', [])
                        if isinstance(related_list, list):
                            for related in related_list:
                                if isinstance(related, str) and (term_lower in related.lower() or related.lower() in term_lower):
                                    score += importance / 2
                scores.append(score)
            
            # Get top-k chunks
            if not scores:
                return chunks[:top_k]  # Fall back to first chunks if no scores
                
            top_indices = np.argsort(scores)[-top_k:][::-1]
            return [chunks[i] for i in top_indices]
        
        except Exception as e:
            print(f"Error finding relevant chunks: {e}")
            traceback.print_exc()
            return chunks[:top_k]  # Fall back to first chunks
    
    def generate_quiz_questions(self, pdf_path: str, num_questions: int = 5) -> List[Dict[str, Any]]:
        """Generate quiz questions from PDF content"""
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            raise ValueError("Could not extract text from PDF")
        
        # Break into chunks and get vibe encodings
        chunks = self.chunk_text(pdf_text)
        print(f"PDF broken into {len(chunks)} chunks")
        
        # Get vibe encodings for each chunk
        vibe_encodings = []
        for i, chunk in enumerate(chunks):
            print(f"Getting vibe encoding for chunk {i+1}/{len(chunks)}...")
            vibe_encoding = self.get_vibe_encoding(chunk)
            vibe_encodings.append(vibe_encoding)
            time.sleep(1)  # Prevent rate limiting
        
        # Generate questions from the whole document
        print(f"Generating {num_questions} quiz questions...")
        questions = []
        
        # First, get topics from the document
        topics_prompt = f"""
        Based on the following text, identify {num_questions} distinct important topics that would be good for quiz questions.
        For each topic, provide a brief description and explain why it's important.
        Format your response as a JSON array of objects with the following fields:
        - topic: the main topic
        - description: brief description
        - importance: why this topic is important to understand
        
        Text to analyze:
        {pdf_text[:5000]}  # Limit text size for the prompt
        """
        
        topics = []
        try:
            topics_response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": topics_prompt}],
                format="json"
            )
            
            topics_content = topics_response['message']['content']
            
            # Extract JSON
            match = re.search(r'```json\s*(.*?)\s*```', topics_content, re.DOTALL)
            if match:
                topics_content = match.group(1)
                
            try:
                topics = json.loads(topics_content)
                if not isinstance(topics, list):
                    # Try to find a JSON array in the text
                    match = re.search(r'\[\s*{.*}\s*\]', topics_content, re.DOTALL)
                    if match:
                        topics_content = match.group(0)
                        topics = json.loads(topics_content)
            except:
                # If parsing fails, generate generic topics
                topics = [{"topic": f"Topic {i+1}", "description": "Important concept from the text"} 
                         for i in range(num_questions)]
        except Exception as e:
            print(f"Error generating topics: {e}")
            traceback.print_exc()
            # Fall back to generic topics
            topics = [{"topic": f"Topic {i+1}", "description": "Important concept from the text"} 
                     for i in range(num_questions)]
            
        if isinstance(topics, dict):
            topics = [topics]
        elif not isinstance(topics, list):
            topics = []
        
        # Now generate questions for each topic
        for i, topic in enumerate(topics[:num_questions]):
            topic_name = topic.get('topic', f"Topic {i+1}")
            description = topic.get('description', '')
            
            print(f"Generating question for topic: {topic_name}")
            
            try:
                # Find relevant chunks for this topic
                relevant_chunks = self.get_relevant_chunks(
                    f"{topic_name} {description}", 
                    chunks, 
                    vibe_encodings, 
                    top_k=3
                )
                
                # Combine relevant chunks
                context = "\n\n".join(relevant_chunks)
                
                # Generate question
                question_prompt = f"""
                Based on this text about "{topic_name}":
                
                {context[:3000]}
                
                Create a multiple-choice quiz question with 4 options and only one correct answer.
                The question should test understanding of important concepts, not trivial details.
                Format your response as a JSON object with these fields:
                - question: the question text
                - options: array of 4 answer choices
                - correct_answer: the letter (A, B, C, or D) of the correct option
                - explanation: detailed explanation of why the correct answer is right and others are wrong
                """
                
                question_response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": question_prompt}],
                    format="json"
                )
                
                question_content = question_response['message']['content']
                
                # Extract JSON
                match = re.search(r'```json\s*(.*?)\s*```', question_content, re.DOTALL)
                if match:
                    question_content = match.group(1)
                
                # Clean up any markdown or extra text
                question_content = re.sub(r'```.*?```', '', question_content, flags=re.DOTALL)
                
                # Try to find a JSON object in the text
                match = re.search(r'\{.*\}', question_content, re.DOTALL)
                if match:
                    question_content = match.group(0)
                
                try:
                    question_data = json.loads(question_content)
                    
                    # Ensure question has required fields
                    if all(k in question_data for k in ['question', 'options', 'correct_answer', 'explanation']):
                        # Standardize options format
                        options = question_data['options']
                        if isinstance(options, list) and len(options) == 4:
                            # Convert options to dict with letter keys if needed
                            options_dict = {
                                'A': options[0],
                                'B': options[1],
                                'C': options[2],
                                'D': options[3]
                            }
                            question_data['options'] = options_dict
                            
                        questions.append(question_data)
                    else:
                        print(f"Question {i+1} missing required fields, skipping")
                except json.JSONDecodeError:
                    print(f"Could not parse question {i+1}, skipping")
            except Exception as e:
                print(f"Error generating question {i+1}: {e}")
                traceback.print_exc()
                
            time.sleep(1)  # Prevent rate limiting
            
        return questions
    
    def evaluate_answers(self, questions: List[Dict[str, Any]], answers: List[str]) -> Dict[str, Any]:
        """
        Evaluate quiz answers and provide explanations
        
        Args:
            questions: List of question objects
            answers: List of answer letters (e.g., ["A", "C", "B", "D"])
            
        Returns:
            Dictionary with score and feedback
        """
        if len(questions) != len(answers):
            raise ValueError(f"Number of answers ({len(answers)}) doesn't match questions ({len(questions)})")
            
        total_questions = len(questions)
        correct_count = 0
        feedback = []
        
        for i, (question, answer) in enumerate(zip(questions, answers)):
            question_text = question.get('question', f'Question {i+1}')
            correct = question.get('correct_answer', '')
            explanation = question.get('explanation', 'No explanation provided')
            
            is_correct = answer.upper() == correct.upper()
            if is_correct:
                correct_count += 1
                
            feedback.append({
                'question_number': i + 1,
                'question': question_text,
                'your_answer': answer.upper(),
                'correct_answer': correct.upper(),
                'is_correct': is_correct,
                'explanation': explanation if not is_correct else "Correct!"
            })
            
        score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        
        return {
            'total_questions': total_questions,
            'correct_count': correct_count,
            'score_percentage': score_percentage,
            'feedback': feedback
        }
    
    def generate_html_quiz(self, questions: List[Dict[str, Any]], output_path: str = "quiz.html"):
        """Generate HTML file with quiz questions"""
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PDF Quiz Generator</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }
                .quiz-container {
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    padding: 20px 30px;
                    margin-bottom: 30px;
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                }
                .question {
                    margin-bottom: 25px;
                    padding-bottom: 15px;
                    border-bottom: 1px solid #e9ecef;
                }
                .question-text {
                    font-weight: 600;
                    font-size: 1.1rem;
                    margin-bottom: 15px;
                    color: #2c3e50;
                }
                .options {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                .option {
                    display: flex;
                    align-items: flex-start;
                    padding: 10px;
                    border-radius: 5px;
                    background-color: #f1f3f5;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }
                .option:hover {
                    background-color: #e9ecef;
                }
                input[type="radio"] {
                    margin-right: 10px;
                    margin-top: 4px;
                }
                label {
                    display: inline-block;
                    cursor: pointer;
                    flex: 1;
                }
                .result-container {
                    display: none;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    padding: 20px 30px;
                }
                .score {
                    text-align: center;
                    font-size: 1.5rem;
                    margin-bottom: 20px;
                    color: #2c3e50;
                }
                .feedback {
                    margin-top: 20px;
                }
                .feedback-item {
                    margin-bottom: 20px;
                    padding: 15px;
                    border-radius: 5px;
                }
                .correct {
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                }
                .incorrect {
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                }
                .explanation {
                    margin-top: 10px;
                    font-style: italic;
                    color: #5a6268;
                }
                button {
                    display: block;
                    width: 200px;
                    padding: 10px;
                    margin: 20px auto;
                    background-color: #4c6ef5;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 1rem;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }
                button:hover {
                    background-color: #3b5bdb;
                }
                .progress {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 20px;
                    color: #6c757d;
                }
                .loader {
                    display: none;
                    text-align: center;
                    margin: 20px 0;
                }
                .loader-spinner {
                    border: 5px solid #f3f3f3;
                    border-top: 5px solid #3498db;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            <h1>PDF Quiz Generator</h1>
            
            <div class="quiz-container" id="quiz-container">
                <div class="progress">
                    <span>Question <span id="current-question">1</span> of <span id="total-questions">0</span></span>
                    <span id="timer">Time: 00:00</span>
                </div>
                
                <div id="questions-container">
                    <!-- Questions will be inserted here -->
                </div>
                
                <div class="loader" id="loader">
                    <div class="loader-spinner"></div>
                    <p>Evaluating your answers...</p>
                </div>
                
                <button id="submit-btn">Submit Answers</button>
            </div>
            
            <div class="result-container" id="result-container">
                <div class="score" id="score-display"></div>
                <div class="feedback" id="feedback-container">
                    <!-- Feedback will be inserted here -->
                </div>
                <button id="retry-btn">Try Again</button>
            </div>
            
            <script>
                // Quiz data
                const quizData = QUIZ_DATA_PLACEHOLDER;
                let userAnswers = {};
                let startTime = new Date();
                let timerInterval;
                
                // DOM elements
                const quizContainer = document.getElementById('quiz-container');
                const questionsContainer = document.getElementById('questions-container');
                const submitBtn = document.getElementById('submit-btn');
                const resultContainer = document.getElementById('result-container');
                const scoreDisplay = document.getElementById('score-display');
                const feedbackContainer = document.getElementById('feedback-container');
                const retryBtn = document.getElementById('retry-btn');
                const currentQuestionSpan = document.getElementById('current-question');
                const totalQuestionsSpan = document.getElementById('total-questions');
                const timerSpan = document.getElementById('timer');
                const loader = document.getElementById('loader');
                
                // Initialize the quiz
                function initQuiz() {
                    userAnswers = {};
                    startTime = new Date();
                    updateTimer();
                    timerInterval = setInterval(updateTimer, 1000);
                    
                    totalQuestionsSpan.textContent = quizData.length;
                    renderQuestions();
                    
                    quizContainer.style.display = 'block';
                    resultContainer.style.display = 'none';
                }
                
                // Update timer display
                function updateTimer() {
                    const now = new Date();
                    const timeDiff = Math.floor((now - startTime) / 1000);
                    const minutes = Math.floor(timeDiff / 60).toString().padStart(2, '0');
                    const seconds = (timeDiff % 60).toString().padStart(2, '0');
                    timerSpan.textContent = `Time: ${minutes}:${seconds}`;
                }
                
                // Render quiz questions
                function renderQuestions() {
                    questionsContainer.innerHTML = '';
                    
                    quizData.forEach((questionData, index) => {
                        const questionDiv = document.createElement('div');
                        questionDiv.className = 'question';
                        questionDiv.dataset.questionIndex = index;
                        
                        const questionText = document.createElement('div');
                        questionText.className = 'question-text';
                        questionText.textContent = `${index + 1}. ${questionData.question}`;
                        
                        const optionsDiv = document.createElement('div');
                        optionsDiv.className = 'options';
                        
                        // Create radio options
                        const optionsLetters = ['A', 'B', 'C', 'D'];
                        let options = questionData.options;
                        
                        // Handle both array and object formats
                        if (Array.isArray(options)) {
                            const tempOptions = {};
                            optionsLetters.forEach((letter, i) => {
                                if (i < options.length) {
                                    tempOptions[letter] = options[i];
                                }
                            });
                            options = tempOptions;
                        }
                        
                        Object.entries(options).forEach(([letter, text]) => {
                            const optionDiv = document.createElement('div');
                            optionDiv.className = 'option';
                            
                            const radioInput = document.createElement('input');
                            radioInput.type = 'radio';
                            radioInput.name = `question-${index}`;
                            radioInput.id = `question-${index}-${letter}`;
                            radioInput.value = letter;
                            
                            const label = document.createElement('label');
                            label.htmlFor = `question-${index}-${letter}`;
                            label.textContent = `${letter}. ${text}`;
                            
                            radioInput.addEventListener('change', () => {
                                userAnswers[index] = letter;
                                updateCurrentQuestion(index + 1);
                            });
                            
                            optionDiv.appendChild(radioInput);
                            optionDiv.appendChild(label);
                            optionsDiv.appendChild(optionDiv);
                        });
                        
                        questionDiv.appendChild(questionText);
                        questionDiv.appendChild(optionsDiv);
                        questionsContainer.appendChild(questionDiv);
                    });
                }
                
                // Update current question display
                function updateCurrentQuestion(number) {
                    currentQuestionSpan.textContent = number;
                }
                
                // Handle quiz submission
                function submitQuiz() {
                    clearInterval(timerInterval);
                    
                    // Show loading indicator
                    loader.style.display = 'block';
                    submitBtn.style.display = 'none';
                    
                    // Convert user answers to array format for evaluation
                    const answersArray = [];
                    for (let i = 0; i < quizData.length; i++) {
                        answersArray.push(userAnswers[i] || '');
                    }
                    
                    // Simulate server evaluation (could be replaced with actual API call)
                    setTimeout(() => {
                        const results = evaluateQuiz(answersArray);
                        displayResults(results);
                        
                        // Hide loader
                        loader.style.display = 'none';
                    }, 1000);
                }
                
                // Evaluate quiz answers
                function evaluateQuiz(answers) {
                    let correctCount = 0;
                    const feedback = [];
                    
                    quizData.forEach((question, index) => {
                        const userAnswer = answers[index];
                        const correctAnswer = question.correct_answer;
                        const isCorrect = userAnswer.toUpperCase() === correctAnswer.toUpperCase();
                        
                        if (isCorrect) {
                            correctCount++;
                        }
                        
                        feedback.push({
                            question_number: index + 1,
                            question: question.question,
                            your_answer: userAnswer.toUpperCase(),
                            correct_answer: correctAnswer.toUpperCase(),
                            is_correct: isCorrect,
                            explanation: isCorrect ? "Correct!" : question.explanation
                        });
                    });
                    
                    const scorePercentage = (correctCount / quizData.length) * 100;
                    
                    return {
                        total_questions: quizData.length,
                        correct_count: correctCount,
                        score_percentage: scorePercentage,
                        feedback: feedback
                    };
                }
                
                // Display quiz results
                function displayResults(results) {
                    // Display score
                    scoreDisplay.textContent = `Score: ${results.correct_count} / ${results.total_questions} (${Math.round(results.score_percentage)}%)`;
                    
                    // Display feedback for each question
                    feedbackContainer.innerHTML = '';
                    
                    results.feedback.forEach(item => {
                        const feedbackDiv = document.createElement('div');
                        feedbackDiv.className = `feedback-item ${item.is_correct ? 'correct' : 'incorrect'}`;
                        
                        const questionPara = document.createElement('p');
                        questionPara.innerHTML = `<strong>Question ${item.question_number}:</strong> ${item.question}`;
                        
                        const answerPara = document.createElement('p');
                        if (item.is_correct) {
                            answerPara.innerHTML = `<strong>Your answer:</strong> ${item.your_answer} ✓`;
                        } else {
                            answerPara.innerHTML = `<strong>Your answer:</strong> ${item.your_answer} ✗<br><strong>Correct answer:</strong> ${item.correct_answer}`;
                        }
                        
                        const explanationPara = document.createElement('p');
                        explanationPara.className = 'explanation';
                        explanationPara.textContent = item.explanation;
                        
                        feedbackDiv.appendChild(questionPara);
                        feedbackDiv.appendChild(answerPara);
                        feedbackDiv.appendChild(explanationPara);
                        feedbackContainer.appendChild(feedbackDiv);
                    });
                    
                    // Show results container
                    quizContainer.style.display = 'none';
                    resultContainer.style.display = 'block';
                }
                
                // Event listeners
                submitBtn.addEventListener('click', submitQuiz);
                retryBtn.addEventListener('click', initQuiz);
                
                // Initialize quiz on page load
                initQuiz();
            </script>
        </body>
        </html>
        """
        
        # Convert questions to JSON for HTML
        questions_json = json.dumps(questions)
        html_content = html_template.replace('QUIZ_DATA_PLACEHOLDER', questions_json)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"Quiz HTML generated at: {output_path}")
        return output_path

# Flask application setup
app = Flask(__name__, static_folder='static')
quiz_generator = None
current_questions = []
upload_folder = 'uploads'
os.makedirs(upload_folder, exist_ok=True)
# Create static folder for quiz output
static_folder = 'static'
os.makedirs(static_folder, exist_ok=True)

@app.route('/')
def index():
    """Render the main page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PDF Quiz Generator</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }
            .container {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 20px 30px;
                margin-bottom: 30px;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            form {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            label {
                font-weight: 600;
                margin-bottom: 5px;
                display: block;
            }
            input, select {
                padding: 10px;
                border: 1px solid #ced4da;
                border-radius: 5px;
                width: 100%;
            }
            button {
                padding: 12px;
                background-color: #4c6ef5;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 1rem;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #3b5bdb;
            }
            .loader {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .loader-spinner {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                color: #dc3545;
                padding: 10px;
                background-color: #f8d7da;
                border-radius: 5px;
                margin-bottom: 15px;
                display: none;
            }
        </style>
    </head>
    <body>
        <h1>PDF Quiz Generator</h1>
        
        <div class="container">
            <div class="error" id="error-container"></div>
            
            <form id="upload-form" enctype="multipart/form-data">
                <div>
                    <label for="pdf-file">Upload PDF:</label>
                    <input type="file" id="pdf-file" name="pdf_file" accept=".pdf" required>
                </div>
                
                <div>
                    <label for="num-questions">Number of Questions:</label>
                    <input type="number" id="num-questions" name="num_questions" value="5" min="1" max="20" required>
                </div>
                
                <div>
                    <label for="model-select">AI Model:</label>
                    <select id="model-select" name="model_name">
                        <option value="llama3.2">Llama 3.2</option>
                        <option value="mistral">Mistral</option>
                        <option value="gemma">Gemma</option>
                    </select>
                </div>
                
                <button type="submit" id="generate-btn">Generate Quiz</button>
            </form>
            
            <div class="loader" id="loader">
                <div class="loader-spinner"></div>
                <p>Generating quiz questions... This may take a few minutes.</p>
            </div>
        </div>
        
        <script>
            document.getElementById('upload-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const errorContainer = document.getElementById('error-container');
                const loader = document.getElementById('loader');
                const generateBtn = document.getElementById('generate-btn');
                
                // Show loader and hide button
                loader.style.display = 'block';
                generateBtn.disabled = true;
                errorContainer.style.display = 'none';
                
                fetch('/generate-quiz', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(data => {
                            throw new Error(data.error || 'Failed to generate quiz');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        window.location.href = data.quiz_url;
                    } else {
                        throw new Error(data.error || 'Failed to generate quiz');
                    }
                })
                .catch(error => {
                    errorContainer.textContent = error.message;
                    errorContainer.style.display = 'block';
                    loader.style.display = 'none';
                    generateBtn.disabled = false;
                });
            });
        </script>
    </body>
    </html>
    """

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    """Generate quiz from uploaded PDF"""
    global quiz_generator, current_questions
    
    try:
        # Check if file was uploaded
        if 'pdf_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
            
        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
            
        # Get parameters
        num_questions = int(request.form.get('num_questions', 5))
        model_name = request.form.get('model_name', 'llama3.2')
        
        # Save uploaded file
        pdf_filename = f"upload_{int(time.time())}_{random.randint(1000, 9999)}.pdf"
        pdf_path = os.path.join(upload_folder, pdf_filename)
        pdf_file.save(pdf_path)
        
        # Initialize quiz generator if needed
        if quiz_generator is None or quiz_generator.model_name != model_name:
            quiz_generator = PDFQuizGenerator(model_name=model_name)
        
        # Generate quiz questions
        current_questions = quiz_generator.generate_quiz_questions(pdf_path, num_questions)
        
        # Generate HTML quiz
        quiz_filename = f"quiz_{int(time.time())}.html"
        output_path = os.path.join('static', quiz_filename)
        quiz_generator.generate_html_quiz(current_questions, output_path)
        
        return jsonify({
            'success': True,
            'quiz_url': f'/quiz/{quiz_filename}'
        })
        
    except Exception as e:
        print(f"Error generating quiz: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/quiz/<filename>')
def serve_quiz(filename):
    """Serve generated quiz"""
    return send_from_directory('static', filename)

@app.route('/api/evaluate', methods=['POST'])
def evaluate_answers():
    """Evaluate quiz answers"""
    global current_questions
    
    try:
        data = request.json
        answers = data.get('answers', [])
        
        if not current_questions:
            return jsonify({'success': False, 'error': 'No quiz questions found'}), 400
            
        # Evaluate answers
        results = quiz_generator.evaluate_answers(current_questions, answers)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"Error evaluating answers: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description='PDF Quiz Generator')
    parser.add_argument('--pdf', type=str, help='Path to PDF file')
    parser.add_argument('--questions', type=int, default=5, help='Number of questions to generate')
    parser.add_argument('--model', type=str, default='llama3.2', help='Ollama model name to use')
    parser.add_argument('--output', type=str, default='quiz.html', help='Output HTML file path')
    parser.add_argument('--web', action='store_true', help='Run as web application')
    
    args = parser.parse_args()
    
    if args.web:
        # Run Flask web application
        print("Starting web application on http://127.0.0.1:5000")
        app.run(debug=True)
    elif args.pdf:
        # Run in CLI mode
        quiz_gen = PDFQuizGenerator(model_name=args.model)
        questions = quiz_gen.generate_quiz_questions(args.pdf, args.questions)
        quiz_gen.generate_html_quiz(questions, args.output)
        print(f"Quiz generated at: {args.output}")
    else:
        print("Please specify --pdf or --web")
        parser.print_help()

if __name__ == "__main__":
    main()