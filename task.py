import streamlit as st
import torch
import os
from dotenv import load_dotenv
from together import Together
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer,DistilBertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification
from datetime import datetime, timedelta
import pandas as pd
from task_css import get_custom_css  # Import the custom CSS function
import gdown

# Set environment variable for offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Load environment variables
load_dotenv()

# Together AI Client with API key from environment variable
client = Together(api_key=os.getenv("TOGETHER_API_KEY", ""))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Intent Model
intent_model_path = "intent_classifier.pth"
# Extract file ID from Google Drive URL
file_id = "1_GDGvV3MVvBguIsjMyDLg3RxUV_gnFAY"
num_intent_labels = 151  # Moved this up before model creation

# Load Emotion Model
emotions_model_path = "./saved_model"
emotions_folder_id = "1gYWkbC_XBw_GZjsfwXvubHFil4BCq_gH"

# Add new pretrained model ID
pretrained_folder_id = "13t_EB2LFhRIwb3dkKDtA0O5NXXZBoG-j"

# Initialize Session State
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False
    st.session_state.models = {}  # Initialize models dict immediately
    st.session_state.tasks = []
    st.session_state.task_counter = 0
    st.session_state.overall_emotion = None
    st.session_state.overall_emotion_label = "Neutral"

# Page Configuration first
st.set_page_config(
    page_title="🚀 AI Productivity Assistant", 
    layout="wide", 
    page_icon="🎯"
)

# Custom CSS for enhanced styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Show loading screen if models aren't ready
if not st.session_state.is_ready:
    st.markdown(
        """
        <div class="loading-container" style="text-align: center; padding: 50px;">
            <div class="loading-spinner"></div>
            <h2>Setting up your AI assistant...</h2>
            <p>This may take a minute. We're downloading the required models.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load models here
    try:
        # First download pretrained models
        if not os.path.exists("pretrained_models"):
            with st.status("Downloading base models...", expanded=True) as status:
                os.makedirs("pretrained_models", exist_ok=True)
                gdown.download_folder(
                    f"https://drive.google.com/drive/folders/{pretrained_folder_id}",
                    output="pretrained_models",
                    quiet=False
                )
                status.update(label="Base models downloaded!", state="complete")

        # Intent Model Loading
        if not os.path.exists(intent_model_path):
            with st.status("Downloading intent model...", expanded=True) as status:
                output = gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    intent_model_path,
                    quiet=False
                )
                status.update(label="Intent model downloaded!", state="complete")

        # Emotion Model Loading
        if not os.path.exists(emotions_model_path):
            with st.status("Downloading emotion model...", expanded=True) as status:
                os.makedirs(emotions_model_path, exist_ok=True)
                gdown.download_folder(
                    f"https://drive.google.com/drive/folders/{emotions_folder_id}",
                    output=emotions_model_path,
                    quiet=False
                )
                status.update(label="Emotion model downloaded!", state="complete")

        # Load and store intent model
        intent_model = AutoModelForSequenceClassification.from_pretrained(
            "pretrained_models/bert-base-uncased",
            num_labels=num_intent_labels,
            ignore_mismatched_sizes=True,  # Add this parameter
            local_files_only=True
        )
        intent_model.load_state_dict(
            torch.load(intent_model_path, map_location=device, weights_only=True)
        )
        st.session_state.models["intent_model"] = intent_model.to(device).eval()
        st.session_state.models["intent_tokenizer"] = AutoTokenizer.from_pretrained(
            "pretrained_models/bert-base-uncased",
            local_files_only=True
        )

        # Load and store emotion model
        emotions_model = AutoModelForSequenceClassification.from_pretrained(
            emotions_model_path,
            ignore_mismatched_sizes=True,  # Add this parameter
            local_files_only=True
        )
        st.session_state.models["emotions_model"] = emotions_model.to(device).eval()
        st.session_state.models["emotions_tokenizer"] = AutoTokenizer.from_pretrained(
            emotions_model_path,
            local_files_only=True
        )

        # Set ready state
        st.session_state.is_ready = True
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Only show main app if models are ready
if st.session_state.is_ready:
    # Title with custom styling
    st.markdown('<div class="main-header">🎯 MoodifyTask: AI Task Prioritization & Wellness Assistant</div>', unsafe_allow_html=True)

    # Emotion Labels
    emotion_label_names = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral"
    ]

    # Emotion Categories
    positive_emotions = ["admiration", "amusement", "approval", "caring", "curiosity", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief", "surprise"]
    negative_emotions = ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"]
    neutral_emotions = ["realization", "neutral"]

    # Predict Intent
    def predict_intent(sentence):
        inputs = st.session_state.models["intent_tokenizer"](
            sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = st.session_state.models["intent_model"](**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        # Mapping Intent IDs to Priorities (0-150)
        PRIORITY_MAPPING = {
            5: [8, 35, 42, 74, 97, 110, 118, 120, 124, 136],  # freeze_account, report_lost_card, flight_status, report_fraud, credit_limit, lost_luggage, dispute_charge, overdraft, cancel_reservation, emergency
            4: [14, 15, 19, 20, 39, 47, 48, 49, 50, 69, 70, 71, 72],  # bill_balance, bill_due, exchange_rate, credit_score, interest_rate, insurance, medical_expenses, appointment_schedule, meeting_schedule, dentist_appointment, doctor_appointment, prescription_refill, pharmacy_hours
            3: [33, 34, 41, 51, 56, 57, 62, 66, 77, 78, 85],  # hotel_reservation, car_rental, restaurant_reservation, tracking_package, check_in, check_out, traffic_update, directions, smart_home_on, smart_home_off, weather_forecast
            2: [0, 1, 3, 6, 9, 13, 16, 17, 21, 25, 27, 28, 36, 40, 45, 52, 61],  # restaurant_reviews, shopping_list, what_song, schedule_meeting, translate, play_music, book_hotel, book_flight, gas_prices, exchange_rate, movie_showtimes, recipe, cancel_flight, book_reservation, order_food, car_services, joke
            1: [2, 4, 5, 7, 10, 11, 12, 18, 22, 23, 24, 26, 30, 31, 32, 37, 38, 43, 44, 46, 53, 54, 55, 58, 59, 60, 63, 64, 65, 67, 68, 73]  
            # tell_joke, fun_fact, trivia, horoscope, dog_fact, cat_fact, define_word, stock_price, sports_update, lottery_results, currency_conversion, holiday_list, language_learning, random_fact, poem, quote, daily_horoscope, joke_request, music_recommendation, podcast_recommendation, celebrity_gossip, movie_recommendation, TV_show_recommendation, book_recommendation, game_recommendation, radio_recommendation, trivia_game, riddle, name_meaning, birthday_reminder, anniversary_reminder, affirmations
        }

        # Find the priority based on predicted_class
        predicted_intent_score = next((priority for priority, ids in PRIORITY_MAPPING.items() if predicted_class in ids), 1)  # Default to 1 if not found

        return predicted_intent_score

    # Emotion to Numeric Score Mapping
    EMOTION_MAPPING = {
        "admiration": 4, "amusement": 3, "anger": 5, "annoyance": 4, "approval": 3,
        "caring": 4, "confusion": 3, "curiosity": 3, "desire": 4, "disappointment": 4,
        "disapproval": 4, "disgust": 5, "embarrassment": 4, "excitement": 5, "fear": 5,
        "gratitude": 3, "grief": 5, "joy": 5, "love": 5, "nervousness": 4,
        "optimism": 4, "pride": 4, "realization": 3, "relief": 3, "remorse": 4,
        "sadness": 5, "surprise": 3, "neutral": 3
    }

    # Function to get numeric emotion score
    def get_emotion_score(emotion):
        return EMOTION_MAPPING.get(emotion.lower(), 3)  # Default to 3 if not found
    # Predict Emotion
    def predict_emotion(sentence):
        if not sentence.strip():
            return 3, "neutral"
        # Ensure the input is a full sentence
        if len(sentence.split()) == 1:  
            sentence = f"I feel {sentence}"
        inputs = st.session_state.models["emotions_tokenizer"](
            sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128
        )
        inputs = {key: val.to(device) for key, val in inputs.items() if key != "token_type_ids"}

        with torch.no_grad():
            outputs = st.session_state.models["emotions_model"](**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        detected_emotion = emotion_label_names[predicted_class]

        # Manually adjust for stress/pressure-related words
        stress_keywords = ["stress", "stressed", "overwhelmed", "pressure", "tense", "burnout"]
        if any(word in sentence.lower() for word in stress_keywords):
            if detected_emotion not in ["sadness", "nervousness"]:
                detected_emotion = "nervousness"  # Change to "sadness" if you prefer

        emotion_score = get_emotion_score(detected_emotion)
        if emotion_score is None:
            emotion_score = 3  # Default neutral score

        return emotion_score, detected_emotion


    # Get Emotion Category
    def get_emotion_category(emotion):
        if emotion in positive_emotions:
            return "positive"
        elif emotion in negative_emotions:
            return "negative"
        else:
            return "neutral"
        

    def normalize_priority(priority, min_value=0, max_value=10):
        return (priority - min_value) / (max_value - min_value)  # Normalize between 0-1

    # Calculate Task Priority
    def calculate_priority_score(predicted_intent_score,emotion_score, emotion,  time_remaining, complexity, emotion_category):
        """
        Calculate an adaptive priority score for tasks based on intent, emotion, time urgency, and complexity.
        """
        emotion_score = emotion_score if emotion_score is not None else 3
        # Normalize time urgency (scale 0 to 1 based on 7 days)
        time_score = max(0, min(1, 1 - (time_remaining.total_seconds() / (7 * 24 * 3600))))

        # Set emotion-based adjustments
        stress_emotions = ["nervousness", "sadness", "fear"]
        frustration_emotions = ["anger", "frustration","disappointment","annoyance"]
        anxiety_emotions = ["anxiety", "uncertainty"]
        

        if emotion_category == "negative":
            if emotion in stress_emotions:
                # Prioritize **easy, quick** tasks to reduce cognitive load
                priority = (predicted_intent_score * 0.15) + (emotion_score * 0.1) + (time_score * 0.3) + ((10 - complexity) * 0.45)
            
            elif emotion in frustration_emotions:
                # Prioritize **engaging** tasks (not too easy) but keep urgency in mind
                priority = (predicted_intent_score * 0.2) + (emotion_score * 0.15) + (time_score * 0.25) + (complexity * 0.4)
            
            elif emotion in anxiety_emotions:
                # Prioritize **urgent, low-complexity** tasks
                priority = (predicted_intent_score * 0.2) + (emotion_score * 0.1) + (time_score * 0.4) + ((10 - complexity) * 0.3)
            
            else:
                # Default for negative emotions: balance urgency and ease
                priority = (predicted_intent_score * 0.2) + (emotion_score * 0.1) + (time_score * 0.3) + ((10 - complexity) * 0.4)

        elif emotion_category == "positive":
            # If the user is in a **good mood**, favor challenging, high-impact tasks
            priority = (predicted_intent_score * 0.35) + (emotion_score * 0.2) + (time_score * 0.25) + (complexity * 0.2)

        else:  # Neutral emotion
            # Keep a balance between difficulty and urgency
            priority = (predicted_intent_score * 0.3) + (emotion_score * 0.2) + (time_score * 0.2) + (complexity * 0.3)

        return normalize_priority(priority)  # Ensure no negative priority values




    # AI-Generated Plan Based on Start Time
    from datetime import datetime

    def get_llama_suggestion(emotion, tasks, selected_datetime):
        """Generate AI plan based on full datetime instead of just time"""
        # Sort tasks by priority (higher priority first)
        sorted_tasks = sorted(tasks, key=lambda x: x["priority_score"], reverse=True)

        # Filter tasks based on selected datetime
        filtered_tasks = [
            task for task in sorted_tasks 
            if task["due_date_time"] >= selected_datetime
        ]

        if not filtered_tasks:
            well_being_prompts = {
                "nervousness": "Suggest mindfulness exercises and short relaxation techniques.",
                "sadness": "Suggest comforting activities like journaling or light exercise.",
                "anger": "Suggest ways to channel frustration productively.",
                "joy": "Suggest ways to maintain productivity while feeling good.",
                "neutral": "Suggest general relaxation activities like listening to music."
            }
            well_being_prompt = f"""
            The user is feeling {emotion}.
            They have no tasks scheduled after {selected_datetime.strftime('%B %d, %I:%M %p')}.
            {well_being_prompts.get(emotion, 'Provide general well-being tips.')}
            """
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": well_being_prompt}],
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    temperature=0.7,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error generating well-being tips: {e}"

        # Prepare the prompt with more detailed datetime information
        task_details = "\n".join([
            f"- {task['description']} (Priority: {task['priority_score']:.2f}, Complexity: {task['complexity']}, Due: {task['due_date_time'].strftime('%B %d, %I:%M %p')})"
            for task in filtered_tasks
        ])

        prompt = f"""
        The user is feeling {emotion}. 
        They need a structured productivity plan starting from {selected_datetime.strftime('%B %d, %I:%M %p')}, not the current time.
        
        Their prioritized tasks (due on or after the selected time), sorted by priority score:
        {task_details}

        Please provide:
        1. A detailed schedule with specific times for each task
        2. Strategic breaks based on task complexity and emotional state
        3. Wellness activities that complement their current emotion
        4. Tips for managing tasks effectively given their emotional state
        5. Suggestions for handling high-priority tasks first while maintaining well-being
        """

        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating AI plan: {e}"


    # Layout with improved spacing
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        # st.markdown('<div class="emotion-analysis">', unsafe_allow_html=True)
        st.markdown('<h3>🌟 Mood Analysis</h3>', unsafe_allow_html=True)
        emotion_sentence = st.text_area(
            "Describe how you're feeling today:", 
            value="", 
            height=150, 
            help="Your emotional state helps us prioritize tasks more effectively"
        )

        if emotion_sentence:
            emotion_score, emotion_label = predict_emotion(emotion_sentence)
            st.session_state.overall_emotion = emotion_score
            st.session_state.overall_emotion_label = emotion_label
            
            st.markdown(f'<div class="emotion-badge">Detected Emotion: {emotion_label}</div>', unsafe_allow_html=True)
            
            # Emotion-based task reprioritization
            for task in st.session_state.tasks:
                task["priority_score"] = calculate_priority_score(
                    task["predicted_intent_score"],
                    emotion_score,
                    emotion_label,
                    task["time_remaining"],
                    task["complexity"],
                    get_emotion_category(emotion_label)
                )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # st.markdown('<div class="task-input">', unsafe_allow_html=True)
        st.markdown('<h3>📅 Add New Task</h3>', unsafe_allow_html=True)
        with st.form("task_form", clear_on_submit=True):
            task_description = st.text_input("Task Description", help="Be specific about what needs to be done")
            col_date, col_time = st.columns(2)
            
            with col_date:
                due_date = st.date_input("Due Date")
            
            with col_time:
                due_time = st.time_input("Due Time")
            
            complexity = st.slider(
                "Task Complexity (1-10)", 
                1, 10, 5, 
                help="Higher complexity may affect task priority"
            )
            
            submitted = st.form_submit_button("➕ Add Task")
            
            if submitted and task_description and due_date and due_time:
                due_date_time = datetime.combine(due_date, due_time)
                time_remaining = due_date_time - datetime.now()
                predicted_intent_score = predict_intent(task_description)
                
                task = {
                    "id": st.session_state.task_counter,  # Add unique ID
                    "description": task_description,
                    "due_date_time": due_date_time,
                    "time_remaining": time_remaining,
                    "complexity": complexity,
                    "predicted_intent_score": predicted_intent_score,
                    "predicted_emotion": st.session_state.overall_emotion,
                    "predicted_label_name": st.session_state.overall_emotion_label,
                    "priority_score": calculate_priority_score(
                        predicted_intent_score, 
                        st.session_state.overall_emotion, 
                        st.session_state.overall_emotion_label, 
                        time_remaining, 
                        complexity, 
                        get_emotion_category(st.session_state.overall_emotion_label)
                    ),
                    "completed": False
                }

                st.session_state.tasks.append(task)
                st.session_state.task_counter += 1  # Increment counter
                st.success("✅ Task Added Successfully!")
        st.markdown('</div>', unsafe_allow_html=True)

    # Task List with Improved Visualization
    if st.session_state.tasks:
        st.markdown('<h3>📌 Task Priority List</h3>', unsafe_allow_html=True)
        
        # Sort tasks by priority
        sorted_tasks = sorted(st.session_state.tasks, key=lambda x: x["priority_score"], reverse=True)
        
        # Create task overview cards
        st.markdown('<div class="task-overview">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(sorted_tasks)}</div><div class="metric-label">Total Tasks</div></div>', unsafe_allow_html=True)
        # with col2:
        #     high_priority = len([t for t in sorted_tasks if t["priority_score"] > 0.7])
        #     st.markdown(f'<div class="metric-card"><div class="metric-value">{high_priority}</div><div class="metric-label">High Priority</div></div>', unsafe_allow_html=True)
        with col2:
            today = datetime.now()
            due_today = len([t for t in sorted_tasks if t["due_date_time"].date() == today.date()])
            st.markdown(f'<div class="metric-card"><div class="metric-value">{due_today}</div><div class="metric-label">Due Today</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display tasks with priority-based styling
        for idx, task in enumerate(sorted_tasks):
            priority_class = "high-priority" if task["priority_score"] > 0.7 else "medium-priority"
            
            # Create a single row for task and buttons
            task_container = st.container()
            with task_container:
                cols = st.columns([0.8, 0.1, 0.1])
                
                # Task content in first column
                with cols[0]:
                    st.markdown(f"""
                        <div class="priority-task {priority_class}">
                            <div class="task-content">
                                <div class="task-header">
                                    <span class="task-title">{task["description"]}</span>
                                    <span class="priority-score">Priority: {task["priority_score"]:.2f}</span>
                                </div>
                                <div class="task-details">
                                    <span class="task-stat">Due: {task["due_date_time"].strftime("%d %b, %I:%M %p")}</span>
                                    <span class="task-stat">Complexity: {task["complexity"]}</span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                st.session_state.editing_task_id = None
                # Edit button
                with cols[1]:
                    if st.button("✏️", key=f"edit_{idx}", help="Edit task"):
                        st.session_state.editing_task_id = idx
                
                # Delete button
                with cols[2]:
                    if st.button("🗑️", key=f"delete_{idx}", help="Delete task"):
                        st.session_state.tasks.pop(idx)
                        st.success("Task deleted!")
                        st.rerun()
                
                # Show edit form below the task if being edited
                if st.session_state.editing_task_id == idx:
                    with st.form(key=f"edit_form_{idx}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            new_description = st.text_input("Description", value=task["description"])
                            new_complexity = st.slider("Complexity", 1, 10, value=task["complexity"])
                        with col2:
                            new_due_date = st.date_input("Due Date", value=task["due_date_time"].date())
                            new_due_time = st.time_input("Due Time", value=task["due_date_time"].time())
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("💾 Save"):
                                # Update task
                                task["description"] = new_description
                                task["due_date_time"] = datetime.combine(new_due_date, new_due_time)
                                task["time_remaining"] = task["due_date_time"] - datetime.now()
                                task["complexity"] = new_complexity
                                
                                # Recalculate priority
                                task["priority_score"] = calculate_priority_score(
                                    task["predicted_intent_score"],
                                    task["predicted_emotion"],
                                    task["predicted_label_name"],
                                    task["time_remaining"],
                                    task["complexity"],
                                    get_emotion_category(task["predicted_label_name"])
                                )
                                st.session_state.editing_task_id = None
                                st.success("Task updated!")
                                st.rerun()
                        
                        with col2:
                            if st.form_submit_button("❌ Cancel"):
                                st.session_state.editing_task_id = None
                                st.rerun()

    # AI Plan Section
    if st.session_state.tasks:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<h3>⏰ AI Task Planning</h3>', unsafe_allow_html=True)
        
        col_date, col_time = st.columns(2)
        
        with col_date:
            plan_date = st.date_input("Select Plan Date", datetime.now().date())
        
        with col_time:
            plan_time = st.time_input("Select Plan Start Time", datetime.now().time())
        
        selected_datetime = datetime.combine(plan_date, plan_time)

        if st.button("📅 Generate AI Plan"):
            suggestion = get_llama_suggestion(
                st.session_state.overall_emotion_label, 
                st.session_state.tasks, 
                selected_datetime  # Pass full datetime object
            )
            st.markdown(f'<div class="info-box">{suggestion}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

