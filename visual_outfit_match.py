import os
import json
import numpy as np
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras.utils import get_custom_objects
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# ==== Custom Layer ====
def l2_norm(x):
    return tf.math.l2_normalize(x, axis=-1)
get_custom_objects().update({'l2_norm': tf.keras.layers.Lambda(l2_norm)})

# ==== Load Model ====
model_path = r"C:\Users\hp\Desktop\outfitmatchai\outfitmatch_model.h5"
model = load_model(model_path, custom_objects={'l2_norm': l2_norm})

# ==== Load Data ====
dataset_folder = r"C:\Users\hp\Desktop\outfitmatchai\data\images"
embeddings_folder = r"C:\Users\hp\Desktop\outfitmatchai\embeddings"
metadata_path = r"C:\Users\hp\Desktop\outfitmatchai\data\polyvore_item_metadata.json"

with open(metadata_path, 'r') as f:
    metadata = json.load(f)
dataset_embeddings = np.load(os.path.join(embeddings_folder, "dataset_embeddings.npy"))
filenames = np.load(os.path.join(embeddings_folder, "filenames.npy"))

filename_to_category = {f"{item_id}.jpg": data.get('semantic_category') for item_id, data in metadata.items()}
filename_to_title = {f"{item_id}.jpg": data.get('title', 'Unknown') for item_id, data in metadata.items()}
target_categories = ['tops', 'bottoms', 'jewellery', 'shoes', 'bags']

# ==== Extract Embedding ====
def extract_embedding(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array)

# ==== Match Outfit ====
def match_outfit(query_path):
    input_embedding = extract_embedding(query_path)
    similarities = cosine_similarity(input_embedding, dataset_embeddings)[0]
    top_items = {}
    for idx in np.argsort(similarities)[::-1]:
        fname = filenames[idx]
        category = filename_to_category.get(fname)
        if category in target_categories and category not in top_items:
            top_items[category] = {
                'filename': fname,
                'score': similarities[idx],
                'title': filename_to_title.get(fname, 'Unknown')
            }
        if len(top_items) == len(target_categories):
            break
    return top_items

# ==== Browse and Show ====
def browse_image():
    path = filedialog.askopenfilename(filetypes=[["Image Files", "*.jpg *.png *.jpeg"]])
    if path:
        show_results(path)

def show_results(img_path):
    global current_outfit, current_image_path
    current_outfit = match_outfit(img_path)
    current_image_path = img_path

    img = Image.open(img_path).resize((150, 150), Image.Resampling.LANCZOS)
    query_img = ImageTk.PhotoImage(img)
    query_label.configure(image=query_img)
    query_label.image = query_img
    query_title.config(text="👗 Your Selected Image")

    for category in target_categories:
        data = current_outfit.get(category)
        label = category_labels[category]
        text = f"{category.title()} \n{data['title']}\nScore: {data['score']:.2f}" if data else f"No {category}"
        label_texts[category].config(text=text)
        if data:
            img = Image.open(os.path.join(dataset_folder, data['filename'])).resize((150, 150), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            label.config(image=tk_img)
            label.image = tk_img
        else:
            label.config(image='')

    rating_frame.pack(pady=10)
    submit_btn.pack(pady=5)

def submit_summary():
    rating = rating_var.get()
    if rating == 0:
        messagebox.showwarning("Missing Rating", "Please rate the outfit before submitting!")
        return
    summary_text = "📋 Suggested Outfit:\n\n"
    for category in target_categories:
        data = current_outfit.get(category)
        if data:
            summary_text += f"• {category.title()}: {data['title']} (Score: {data['score']:.2f})\n"
        else:
            summary_text += f"• {category.title()}: Not Found\n"
    summary_text += f"\n⭐ Your Rating: {rating} Stars"
    summary_textbox.delete("1.0", tk.END)
    summary_textbox.insert(tk.END, summary_text)
    messagebox.showinfo("Thanks!", f"Thank you for rating this outfit {rating} ⭐")

# ==== Login ====
def login():
    if username_entry.get() == "admin" and password_entry.get() == "1234":
        login_frame.destroy()
        launch_main_screen()
    else:
        messagebox.showerror("Login Failed", "Incorrect username or password!")

# ==== Launch Main UI ====
def launch_main_screen():
    root.geometry("1280x750")
    root.title("🖤💛 OutfitMatch AI - Stylish Companion")

    # Background image
    bg_img_main = Image.open(r"C:\Users\hp\Desktop\bgout.jpg").resize((1280, 750), Image.Resampling.LANCZOS)
    bg_photo_main = ImageTk.PhotoImage(bg_img_main)
    bg_label_main = tk.Label(root, image=bg_photo_main)
    bg_label_main.image = bg_photo_main
    bg_label_main.place(relx=0, rely=0, relwidth=1, relheight=1)
    bg_label_main.lower()

    # Main container frame with centered content
    main_container = tk.Frame(root, bg="#f8f8f8")
    main_container.pack(fill="both", expand=True)

    # Canvas and scrollbar with centered content
    canvas = tk.Canvas(main_container, bg="#f8f8f8", highlightthickness=0)
    scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
    
    # Center frame to hold all content
    center_frame = tk.Frame(canvas, bg="#f8f8f8")
    canvas.create_window((root.winfo_width()//2, 0), window=center_frame, anchor="n")
    
    # Configure canvas scrolling
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        # Keep content centered when resizing
        canvas.itemconfig(1, anchor="n", width=event.width)
    
    center_frame.bind("<Configure>", on_frame_configure)
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Pack canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Content container with fixed width to center everything
    content_frame = tk.Frame(center_frame, bg="#f8f8f8", padx=20, pady=20)
    content_frame.pack()

    # Logo
    logo_img = Image.open(r"C:\Users\hp\Desktop\outfitlogo.jpg").resize((100, 100), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(content_frame, image=logo_photo, bg="#f8f8f8")
    logo_label.image = logo_photo
    logo_label.pack(pady=10)

    # Title
    tk.Label(content_frame, text="OutfitMatch AI", font=("Impact", 32), fg="black", bg="#f8f8f8").pack(pady=5)
    
    # Upload button
    ttk.Button(content_frame, text="📸 Upload Your Outfit", command=browse_image).pack(pady=20)

    # Results frame
    global frame, query_label, query_title, category_labels, label_texts, rating_frame, rating_var, summary_textbox, submit_btn
    frame = tk.Frame(content_frame, bg="#f8f8f8")
    frame.pack(pady=20)

    # Query image display - centered
    query_title = tk.Label(frame, text="", font=("Segoe UI", 12, "bold"), bg="#f8f8f8", fg="black")
    query_title.grid(row=0, column=0, columnspan=len(target_categories), pady=10)
    
    query_label = tk.Label(frame, bg="#f8f8f8")
    query_label.grid(row=1, column=0, columnspan=len(target_categories), pady=10)

    # Category displays - centered
    category_labels = {}
    label_texts = {}
    for i, category in enumerate(target_categories):
        cat_label = tk.Label(frame, bg="#f8f8f8")
        cat_label.grid(row=2, column=i, padx=10, pady=5)
        
        txt_label = tk.Label(frame, text="", font=("Segoe UI", 10), bg="#f8f8f8", 
                           fg="black", wraplength=150, justify='center')
        txt_label.grid(row=3, column=i, padx=10, pady=5)
        
        category_labels[category] = cat_label
        label_texts[category] = txt_label

    # Rating section - centered
    rating_frame = tk.Frame(content_frame, bg="#f8f8f8")
    tk.Label(rating_frame, text="⭐ Rate This Outfit:", font=("Segoe UI", 14, "bold"), 
            bg="#f8f8f8", fg="black").pack(side='left', padx=5)
    
    rating_var = tk.IntVar()
    for i in range(1, 6):
        tk.Radiobutton(rating_frame, text=f"{i}⭐", variable=rating_var, value=i,
                     font=("Segoe UI", 12), bg="#f8f8f8", fg="black", 
                     activebackground="#f8f8f8", activeforeground="gold", 
                     selectcolor="gold").pack(side='left', padx=5)
    rating_frame.pack(pady=10)

    # Summary section - centered with fixed width
    summary_container = tk.Frame(content_frame, bg="#f8f8f8")
    summary_container.pack(pady=20)
    
    tk.Label(summary_container, text="Outfit Summary:", font=("Segoe UI", 14, "bold"),
            bg="#f8f8f8", fg="black").pack(anchor='w', pady=5)
    
    inner_frame = tk.Frame(summary_container, bg="#f8f8f8")
    inner_frame.pack()
    
    inner_scrollbar = ttk.Scrollbar(inner_frame)
    inner_scrollbar.pack(side='right', fill='y')
    
    summary_textbox = tk.Text(inner_frame, font=("Consolas", 12), bg="white", 
                            fg="black", width=80, height=8, relief='sunken', bd=3, 
                            wrap='word', yscrollcommand=inner_scrollbar.set)
    summary_textbox.pack(side='left', fill='both', expand=True)
    inner_scrollbar.config(command=summary_textbox.yview)

    # Submit button - centered
    submit_btn = ttk.Button(content_frame, text="✅ Submit Feedback", command=submit_summary)
    submit_btn.pack(pady=20)

    # Make sure content stays centered when window resizes
    def on_resize(event):
        canvas.itemconfig(1, anchor="n", width=event.width)
    
    canvas.bind("<Configure>", on_resize)

# ==== App Start ====
root = tk.Tk()
root.geometry("600x400")
root.title("Login - OutfitMatch AI")
root.configure(bg="#f8f8f8")

# Background for login
bg_img = Image.open(r"C:\Users\hp\Desktop\bgout.jpg").resize((1600, 1400), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_img)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)
bg_label.image = bg_photo
bg_label.lower()

# Login frame - centered
login_frame = tk.Frame(root, bg="#f8f8f8", padx=30, pady=30)
login_frame.place(relx=0.5, rely=0.5, anchor='center')

tk.Label(login_frame, text="🖤💛 OutfitMatch AI Login", font=("Impact", 28), bg="#f8f8f8", fg="black").pack(pady=20)
tk.Label(login_frame, text="Username:", font=("Segoe UI", 14), bg="#f8f8f8", fg="black").pack()
username_entry = ttk.Entry(login_frame, font=("Segoe UI", 12))
username_entry.pack(pady=5)
tk.Label(login_frame, text="Password:", font=("Segoe UI", 14), bg="#f8f8f8", fg="black").pack()
password_entry = ttk.Entry(login_frame, font=("Segoe UI", 12), show="*")
password_entry.pack(pady=5)
ttk.Button(login_frame, text="🔐 Login", command=login).pack(pady=20)

root.mainloop()