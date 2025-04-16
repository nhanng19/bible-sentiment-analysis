# ========================
# IMPORTS
# ========================
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tkinter as tk
from tkinter import ttk, messagebox
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ========================
# CONSTANTS
# ========================
# Book number to book name mapping
BOOK_NAMES = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
    "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations",
    "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk",
    "Zephaniah", "Haggai", "Zechariah", "Malachi", "Matthew",
    "Mark", "Luke", "John", "Acts", "Romans",
    "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians",
    "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy",
    "Titus", "Philemon", "Hebrews", "James", "1 Peter",
    "2 Peter", "1 John", "2 John", "3 John", "Jude",
    "Revelation"
]

EMOTION_COLORS = {
    'Joy': '#4CAF50',       # Green
    'Trust': '#8BC34A',     # Light Green
    'Fear': '#9C27B0',      # Purple
    'Surprise': '#FF9800',  # Orange
    'Sadness': '#2196F3',   # Blue
    'Disgust': '#795548',   # Brown
    'Anger': '#F44336',     # Red
    'Anticipation': '#FFC107' # Yellow
}

# ========================
# DATA PROCESSING FUNCTIONS
# ========================
def load_and_prepare_bible(path):
    """Load and clean the Bible dataset"""
    df = pd.read_csv(path)
    df.columns = ['id', 'b', 'c', 'v', 't']  # Rename columns
    df['Book'] = df['b'].apply(lambda x: BOOK_NAMES[x - 1])  # Map book number to name
    df.rename(columns={'c': 'Chapter', 'v': 'Verse', 't': 'Text'}, inplace=True)
    return df[['Book', 'Chapter', 'Verse', 'Text']]

def load_nrc_lexicon(path="NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"):
    """Load NRC Emotion Lexicon from file"""
    lexicon = defaultdict(dict)
    if not os.path.exists(path):
        print(f"NRC Lexicon file not found at {path}. Falling back to VADER sentiment analysis.")
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                word, emotion, value = parts
                lexicon[word][emotion] = int(value)
    return lexicon

def analyze_emotions(text, lexicon):
    """Analyze text emotions using NRC lexicon"""
    emotions = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
    emotion_scores = {e: 0 for e in emotions}
    
    try:
        words = nltk.word_tokenize(text.lower())
        for word in words:
            if word in lexicon:
                for emotion in emotions:
                    if emotion in lexicon[word]:
                        emotion_scores[emotion] += lexicon[word][emotion]
        return emotion_scores
    except:
        return emotion_scores

def analyze_sentiments(df):
    """Perform sentiment analysis combining VADER and NRC"""
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    sia = SentimentIntensityAnalyzer()
    
    # Get VADER sentiment scores
    sentiments = df['Text'].apply(lambda text: sia.polarity_scores(str(text)))
    sentiment_df = pd.DataFrame(list(sentiments))
    
    # Get NRC emotion scores if available
    if nrc_lexicon:
        emotion_scores = df['Text'].apply(lambda text: analyze_emotions(str(text), nrc_lexicon))
        emotion_df = pd.DataFrame(list(emotion_scores))
        return pd.concat([df, sentiment_df, emotion_df], axis=1)
    else:
        return pd.concat([df, sentiment_df], axis=1)

# ========================
# ANALYSIS FUNCTIONS
# ========================
def get_top_verses(emotion):
    """Get top verses for a specific emotion"""
    emotion_map = {
        'Joy': 'joy',
        'Trust': 'trust',
        'Fear': 'fear',
        'Surprise': 'surprise',
        'Sadness': 'sadness',
        'Disgust': 'disgust',
        'Anger': 'anger',
        'Anticipation': 'anticipation'
    }
    
    nrc_emotion = emotion_map.get(emotion)
    
    if nrc_lexicon and nrc_emotion in analyzed_df.columns:
        # Use NRC emotion scores if available
        sorted_df = analyzed_df.sort_values(by=nrc_emotion, ascending=False)
    else:
        # Fall back to VADER sentiment
        sentiment_map = {
            'Joy': 'pos', 'Trust': 'pos',
            'Fear': 'neg', 'Surprise': 'neu',
            'Sadness': 'neg', 'Disgust': 'neg',
            'Anger': 'neg', 'Anticipation': 'pos'
        }
        sentiment_col = sentiment_map.get(emotion, 'neu')
        sorted_df = analyzed_df.sort_values(by=sentiment_col, ascending=False)
    
    return sorted_df.head(10)[['Book', 'Chapter', 'Verse', 'Text']]

def rank_books():
    """Rank books by average sentiment"""
    book_sentiments = analyzed_df.groupby(['Book']).mean(numeric_only=True)
    return book_sentiments[['compound']].sort_values(by='compound', ascending=False)

# ========================
# VISUALIZATION FUNCTIONS
# ========================
def show_emotion_pie_chart():
    """Show pie chart of emotion distribution"""
    if not nrc_lexicon:
        messagebox.showwarning("No NRC Lexicon", "NRC Emotion Lexicon not found. Cannot show emotion distribution.")
        return
    
    emotions = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
    emotion_totals = analyzed_df[emotions].sum()
    
    win = tk.Toplevel(root)
    win.title("Bible Emotion Distribution")
    win.geometry("600x600")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {
        'joy': '#4CAF50',
        'trust': '#8BC34A',
        'fear': '#9C27B0',
        'surprise': '#FF9800',
        'sadness': '#2196F3',
        'disgust': '#795548',
        'anger': '#F44336',
        'anticipation': '#FFC107'
    }
    
    wedges, texts, autotexts = ax.pie(
        emotion_totals,
        labels=[e.capitalize() for e in emotions],
        autopct='%1.1f%%',
        startangle=140,
        colors=[colors[e] for e in emotions]
    )
    
    ax.axis('equal')
    ax.set_title('Emotion Distribution in the Bible', pad=20)
    
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def show_book_rankings():
    """Show bar graph of book rankings"""
    rankings = rank_books()
    
    win = tk.Toplevel(root)
    win.title("Book Sentiment Rankings")
    win.geometry("1000x800")
    
    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = range(len(rankings))
    
    ax.barh(
        y_pos, 
        rankings['compound'], 
        align='center',
        color='#2196F3'
    )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rankings.index, fontsize=6, fontweight='light')
    ax.invert_yaxis()
    ax.set_xlabel('Average Sentiment Score')
    ax.set_title('Bible Books Ranked by Average Sentiment')
    
    for i, v in enumerate(rankings['compound']):
        ax.text(v, i, f" {v:.2f}", color='black', va='center', fontsize=6)
    
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def show_sentiment_heatmap():
    """Show sentiment heatmap by book and chapter"""
    heatmap_data = analyzed_df.groupby(['Book', 'Chapter'])['compound'].mean().unstack()
    
    win = tk.Toplevel(root)
    win.title("Bible Sentiment Heatmap")
    win.geometry("1000x800")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks([])
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=6, fontweight='light')
    ax.set_title("Sentiment Heatmap by Book and Chapter", pad=20)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Sentiment Score (Compound)', rotation=270, labelpad=20)
    
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ========================
# GUI FUNCTIONS
# ========================
def on_emotion_selected(emotion):
    """Handle emotion selection from GUI"""
    if emotion:
        top_verses = get_top_verses(emotion)
        result_text.delete(1.0, tk.END)
        
        if not nrc_lexicon:
            result_text.insert(tk.END, "Note: Using basic sentiment analysis (NRC lexicon not found)\n\n")
        
        for _, verse in top_verses.iterrows():
            ref = f"{verse['Book']} {verse['Chapter']}:{verse['Verse']}"
            result_text.insert(tk.END, f"{ref}\n{verse['Text']}\n\n")
    else:
        messagebox.showwarning("No Emotion", "Please select how you're feeling.")

def create_emotion_button(emotion):
    """Create an emotion button with appropriate styling"""
    button = tk.Button(
        emotion_buttons_frame, 
        text=emotion, 
        command=lambda: on_emotion_selected(emotion),
        font=("Helvetica", 12), 
        bg=EMOTION_COLORS.get(emotion, '#4CAF50'), 
        fg="white", 
        relief="flat", 
        padx=8, 
        pady=4
    )
    button.grid(row=0, column=EMOTIONS.index(emotion), padx=5, pady=5, ipadx=10, ipady=5)

# ========================
# MAIN PROGRAM
# ========================
if __name__ == "__main__":
    # Load data and initialize analysis
    nrc_lexicon = load_nrc_lexicon()
    bible_df = load_and_prepare_bible("bible.csv")
    analyzed_df = analyze_sentiments(bible_df)

    # Initialize GUI
    root = tk.Tk()
    root.title("Bible Emotion Analysis")
    root.geometry("600x700")
    root.config(bg="#f7f7f7")

    # Header
    header_label = tk.Label(root, text="How are you feeling today?", font=("Helvetica", 16, "bold"), bg="#f7f7f7")
    header_label.pack(pady=10)

    # Emotion buttons
    EMOTIONS = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
    emotion_buttons_frame = tk.Frame(root, bg="#f7f7f7")
    emotion_buttons_frame.pack(pady=10)
    
    for emotion in EMOTIONS:
        create_emotion_button(emotion)

    # Results display
    result_frame = tk.Frame(root, bg="#f7f7f7")
    result_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    result_text = tk.Text(
        result_frame, 
        wrap=tk.WORD, 
        height=15, 
        width=70, 
        font=("Helvetica", 10), 
        bg="#e8e8e8", 
        fg="#333",
        padx=10,
        pady=10
    )
    scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=result_text.yview)
    result_text.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    result_text.pack(side="left", fill=tk.BOTH, expand=True)

    # Visualization buttons
    button_frame = tk.Frame(root, bg="#f7f7f7")
    button_frame.pack(pady=10)

    tk.Button(
        button_frame, 
        text="Show Book Rankings", 
        command=show_book_rankings, 
        font=("Helvetica", 12), 
        bg="#2196F3", 
        fg="white"
    ).pack(side="left", padx=5)

    tk.Button(
        button_frame, 
        text="Show Emotion Distribution", 
        command=show_emotion_pie_chart, 
        font=("Helvetica", 12), 
        bg="#9C27B0", 
        fg="white"
    ).pack(side="left", padx=5)

    tk.Button(
        button_frame, 
        text="Show Heatmap", 
        command=show_sentiment_heatmap, 
        font=("Helvetica", 12), 
        bg="#FF5722", 
        fg="white"
    ).pack(side="left", padx=5)

    # Status bar
    status_bar = tk.Label(
        root, 
        text="NRC Emotion Lexicon loaded" if nrc_lexicon else "Using basic sentiment analysis (NRC lexicon not found)",
        bd=1, 
        relief=tk.SUNKEN, 
        anchor=tk.W,
        bg="#f7f7f7"
    )
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()