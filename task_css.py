def get_custom_css():
    return """
    <style>
    :root {
        /* Refined Color Palette */
        --primary-blue: #3B82F6;        /* Vibrant Blue */
        --primary-dark: #1E40AF;        /* Deeper Blue */
        --accent-teal: #0EA5E9;         /* Bright Teal */
        --background-light: #F9FAFB;    /* Soft White */
        --text-dark: #1E293B;           /* Deep Navy */
        --text-medium: #475569;         /* Medium Slate */
        --accent-orange: #F97316;       /* Warm Orange */
        --success-green: #10B981;       /* Emerald Green */
        --warning-yellow: #FBBF24;      /* Amber Yellow */
        --error-red: #EF4444;           /* Cherry Red */
        
        /* Refined Gradients */
        --gradient-primary: linear-gradient(135deg, var(--primary-blue), var(--primary-dark));
        --gradient-accent: linear-gradient(135deg, var(--accent-teal), #38BDF8);
        --gradient-warm: linear-gradient(135deg, var(--accent-orange), #FB923C);
    }

    /* Global Reset with Professional Typography */
    body, .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background-color: var(--background-light) !important;
        color: var(--text-dark);
        line-height: 1.6;
        letter-spacing: -0.011em;
    }

    /* App Container with Refined Depth */
    [data-testid="stAppViewContainer"] {
        background-color: var(--background-light) !important;
        max-width: 1100px;
        margin: 0 auto;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 
            0 20px 25px -5px rgba(59, 130, 246, 0.1),
            0 10px 10px -5px rgba(59, 130, 246, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
    }

    /* Professional Header */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
        color: var(--primary-blue);
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .main-header::before {
        
        margin-right: 15px;
        font-size: 2.2rem;
        transition: transform 0.3s ease;
    }

    .main-title:hover::before {
        transform: scale(1.1) rotate(5deg);
    }

    /* Professional Card Sections */
    .emotion-analysis, .task-input {
        background-color: white;
        border-radius: 12px;
        padding: 1.8rem;
        box-shadow: 
            0 4px 6px -1px rgba(59, 130, 246, 0.1),
            0 2px 4px -1px rgba(59, 130, 246, 0.06);
        margin-bottom: 1.5rem;
        transition: all 0.2s ease;
        border-top: 3px solid var(--primary-blue);
    }

    .emotion-analysis:hover, .task-input:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 10px 15px -3px rgba(59, 130, 246, 0.1),
            0 4px 6px -2px rgba(59, 130, 246, 0.05);
    }

    /* Section Headers */
    .stMarkdown h3 {
        color: var(--primary-blue);
        font-weight: 600;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
        border-bottom: 1px solid rgba(59, 130, 246, 0.2);
        padding-bottom: 0.5rem;
    }

    /* Polished Input Elements */
    .stTextArea textarea, 
    .stTextInput>div>div>input {
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
        background-color: white !important;
        color: var(--text-dark) !important;
        font-weight: 400;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }

    .stTextArea textarea:focus, 
    .stTextInput>div>div>input:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 
            0 0 0 3px rgba(59, 130, 246, 0.15) !important,
            0 1px 2px rgba(0, 0, 0, 0.05) !important;
        outline: none !important;
    }

    /* Contextual Badges */
    .emotion-badge {
        background: var(--gradient-accent);
        color: white !important;
        border-radius: 6px;
        padding: 8px 12px;
        font-weight: 600;
        display: inline-block;
        margin-top: 10px;
        box-shadow: 0 2px 4px rgba(14, 165, 233, 0.2);
    }

    .warning-badge {
        background: var(--gradient-warm);
        color: white !important;
        border-radius: 6px;
        padding: 8px 12px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(249, 115, 22, 0.2);
    }

    /* Professional Button */
    .stButton>button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        transition: all 0.2s ease !important;
        box-shadow: 
            0 4px 6px -1px rgba(59, 130, 246, 0.2),
            0 2px 4px -1px rgba(59, 130, 246, 0.1);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 6px 10px -1px rgba(59, 130, 246, 0.25),
            0 4px 6px -1px rgba(59, 130, 246, 0.15);
    }

    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 
            0 2px 4px -1px rgba(59, 130, 246, 0.2),
            0 1px 2px -1px rgba(59, 130, 246, 0.1);
    }

    /* Improved Slider */
    .stSlider {
        margin-top: 12px;
    }

    .stSlider > div > div > div {
        background-color: #CBD5E1 !important;
        height: 6px !important;
        border-radius: 3px !important;
    }

    .stSlider > div > div > div > div {
        background: var(--primary-blue) !important;
        box-shadow: 0 0 0 2px white, 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        width: 18px !important;
        height: 18px !important;
        border-radius: 50% !important;
        transition: transform 0.2s ease !important;
    }

    .stSlider > div > div > div > div:hover {
        transform: scale(1.15) !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background-color: var(--primary-blue) !important;
        border-radius: 4px !important;
    }

    /* Select Boxes */
    .stSelectbox label {
        color: var(--text-medium) !important;
        font-weight: 500 !important;
    }

    .stSelectbox > div > div > div {
        background-color: white !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 8px !important;
        padding: 4px 8px !important;
    }

    /* Checkbox */
    .stCheckbox label {
        color: var(--text-medium) !important;
        font-size: 0.95rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: rgba(59, 130, 246, 0.1) !important;
        border-radius: 8px !important;
        padding: 2px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        border: none !important;
        color: var(--text-medium) !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: var(--primary-blue) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* Info Boxes */
    .info-box {
        background-color: rgba(14, 165, 233, 0.1);
        border-left: 3px solid var(--accent-teal);
        border-radius: 6px;
        padding: 15px;
        margin: 15px 0;
        color: var(--text-dark);
    }
    
    .success-box {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 3px solid var(--success-green);
        border-radius: 6px;
        padding: 15px;
        margin: 15px 0;
    }
    
    .warning-box {
        background-color: rgba(251, 191, 36, 0.1);
        border-left: 3px solid var(--warning-yellow);
        border-radius: 6px;
        padding: 15px;
        margin: 15px 0;
    }
    
    .error-box {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 3px solid var(--error-red);
        border-radius: 6px;
        padding: 15px;
        margin: 15px 0;
    }

    /* Data Elements */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        align-items: center;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        border-top: 3px solid var(--primary-blue);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-blue);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-medium);
        margin-top: 5px;
    }

    /* Action Menu */
    .action-menu {
        position: relative;
        display: inline-block;
    }

    .action-menu-content {
        display: none;
        position: absolute;
        right: 0;
        background-color: white;
        min-width: 120px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        z-index: 1;
        border-radius: 8px;
        overflow: hidden;
    }

    .action-menu-content a {
        color: var(--text-dark);
        padding: 12px 16px;
        text-decoration: none;
        display: block;
        transition: background-color 0.2s ease;
    }

    .action-menu-content a:hover {
        background-color: var(--primary-blue);
        color: white;
    }

    .action-menu:hover .action-menu-content {
        display: block;
    }

    .action-menu .three-dots {
        cursor: pointer;
        font-size: 1.5rem;
        color: var(--text-medium);
    }

    /* Task Content Styling */
    .task-content {
        flex: 1;
        padding-right: 20px;
    }

    .task-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .task-title {
        font-weight: 600;
        color: var(--text-dark);
        font-size: 1rem;
    }

    .priority-score {
        background: var(--gradient-primary);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .task-details {
        display: flex;
        gap: 16px;
    }

    .task-stat {
        color: var(--text-medium);
        font-size: 0.9rem;
    }

    /* Priority Task List */
    .priority-task {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .high-priority {
        border-left: 4px solid var(--error-red);
    }

    .medium-priority {
        border-left: 4px solid var(--warning-yellow);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        [data-testid="stAppViewContainer"] {
            padding: 1.2rem;
            border-radius: 12px;
        }
        
        .main-title {
            font-size: 1.8rem;
        }
        
        .emotion-analysis, .task-input {
            padding: 1.2rem;
        }
        
        .metric-value {
            font-size: 1.6rem;
        }
    }
    </style>
    """