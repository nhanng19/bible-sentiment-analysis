import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tkinter as tk
from tkinter import ttk, messagebox
from collections import defaultdict
import os

# Book number to book name mapping
book_names = [
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

# Load CSV and clean
def load_and_prepare_bible(path):
    df = pd.read_csv(path)
    df.columns = ['id', 'b', 'c', 'v', 't']  # Rename columns
    df['Book'] = df['b'].apply(lambda x: book_names[x - 1])  # Map book number to name
    df.rename(columns={'c': 'Chapter', 'v': 'Verse', 't': 'Text'}, inplace=True)
    return df[['Book', 'Chapter', 'Verse', 'Text']]

# Load NRC Emotion Lexicon
def load_nrc_lexicon(path="NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"):
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

# Analyze emotions using NRC lexicon
def analyze_emotions(text, lexicon):
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

# Sentiment analysis combining VADER and NRC
def analyze_sentiments(df):
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

# Get top verses for emotion
def get_top_verses(emotion):
    # Map GUI emotion names to NRC lexicon names
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

# Rank chapters by compound sentiment
def rank_chapters():
    chapter_sentiments = analyzed_df.groupby(['Book', 'Chapter']).mean(numeric_only=True)
    return chapter_sentiments[['compound']].sort_values(by='compound', ascending=False)

# GUI callbacks
def on_emotion_selected(emotion):
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

def show_chapter_rankings():
    rankings = rank_chapters()
    win = tk.Toplevel(root)
    win.title("Chapter Sentiment Rankings")
    win.geometry("400x500")
    text = tk.Text(win, wrap=tk.WORD)
    scroll = ttk.Scrollbar(win, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    text.pack(expand=True, fill=tk.BOTH)
    for (book, chapter), row in rankings.head(50).iterrows():
        text.insert(tk.END, f"{book} {chapter}: {row['compound']:.2f}\n")

# Load NRC lexicon
nrc_lexicon = load_nrc_lexicon()

# Load and analyze Bible data
bible_df = load_and_prepare_bible("bible.csv")
analyzed_df = analyze_sentiments(bible_df)

# GUI setup
root = tk.Tk()
root.title("Bible Emotion Analysis")
root.geometry("600x700")
root.config(bg="#f7f7f7")

# Header Label
header_label = tk.Label(root, text="How are you feeling today?", font=("Helvetica", 16, "bold"), bg="#f7f7f7")
header_label.pack(pady=10)

# Emotion Pills
emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
emotion_buttons_frame = tk.Frame(root, bg="#f7f7f7")
emotion_buttons_frame.pack(pady=10)

def create_emotion_button(emotion):
    color = {
        'Joy': '#4CAF50',       # Green
        'Trust': '#8BC34A',     # Light Green
        'Fear': '#9C27B0',      # Purple
        'Surprise': '#FF9800',   # Orange
        'Sadness': '#2196F3',    # Blue
        'Disgust': '#795548',    # Brown
        'Anger': '#F44336',      # Red
        'Anticipation': '#FFC107' # Yellow
    }.get(emotion, '#4CAF50')
    
    button = tk.Button(
        emotion_buttons_frame, 
        text=emotion, 
        command=lambda: on_emotion_selected(emotion),
        font=("Helvetica", 12), 
        bg=color, 
        fg="white", 
        relief="flat", 
        padx=10, 
        pady=5
    )
    button.grid(row=0, column=emotions.index(emotion), padx=5, pady=5, ipadx=10, ipady=5)

for emotion in emotions:
    create_emotion_button(emotion)

# Chatbox-style result display
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

# Button for showing chapter rankings
button_frame = tk.Frame(root, bg="#f7f7f7")
button_frame.pack(pady=10)

tk.Button(
    button_frame, 
    text="Show Chapter Rankings", 
    command=show_chapter_rankings, 
    font=("Helvetica", 12), 
    bg="#2196F3", 
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