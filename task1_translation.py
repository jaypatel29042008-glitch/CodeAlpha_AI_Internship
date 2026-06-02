"""
TASK 1: Language Translation Tool
CodeAlpha AI Internship
Uses Google Translate API (googletrans) with a Tkinter UI
"""

import tkinter as tk
from tkinter import ttk, messagebox
from googletrans import Translator, LANGUAGES

# ── Language map ──────────────────────────────────────────────────────────────
LANG_OPTIONS = {v.title(): k for k, v in LANGUAGES.items()}
LANG_NAMES   = sorted(LANG_OPTIONS.keys())

translator = Translator()

# ── Core translation logic ────────────────────────────────────────────────────
def translate_text():
    source_text = input_box.get("1.0", tk.END).strip()
    if not source_text:
        messagebox.showwarning("Input Error", "Please enter text to translate.")
        return

    src_lang  = LANG_OPTIONS.get(src_combo.get(), "auto")
    dest_lang = LANG_OPTIONS.get(dst_combo.get(), "en")

    try:
        result = translator.translate(source_text, src=src_lang, dest=dest_lang)
        output_box.config(state="normal")
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, result.text)
        output_box.config(state="disabled")
        detected_label.config(
            text=f"Detected Language: {LANGUAGES.get(result.src, result.src).title()}"
        )
    except Exception as e:
        messagebox.showerror("Translation Error", str(e))


def copy_translation():
    text = output_box.get("1.0", tk.END).strip()
    if text:
        root.clipboard_clear()
        root.clipboard_append(text)
        messagebox.showinfo("Copied", "Translation copied to clipboard!")


def clear_all():
    input_box.delete("1.0", tk.END)
    output_box.config(state="normal")
    output_box.delete("1.0", tk.END)
    output_box.config(state="disabled")
    detected_label.config(text="Detected Language: —")

# ── GUI setup ──────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("🌐 Language Translation Tool — CodeAlpha")
root.geometry("750x560")
root.resizable(False, False)
root.configure(bg="#1e1e2e")

FONT_H1   = ("Segoe UI", 16, "bold")
FONT_LBL  = ("Segoe UI", 11)
FONT_TEXT = ("Consolas", 11)
BG        = "#1e1e2e"
FG        = "#cdd6f4"
ACCENT    = "#89b4fa"
BOX_BG    = "#313244"

# Title
tk.Label(root, text="🌐 Language Translation Tool", font=FONT_H1,
         bg=BG, fg=ACCENT).pack(pady=(18, 4))
tk.Label(root, text="Powered by Google Translate API  |  CodeAlpha AI Internship",
         font=("Segoe UI", 9), bg=BG, fg="#6c7086").pack()

# Language selectors
sel_frame = tk.Frame(root, bg=BG)
sel_frame.pack(pady=12)

tk.Label(sel_frame, text="Source Language:", font=FONT_LBL, bg=BG, fg=FG).grid(
    row=0, column=0, padx=8)
src_combo = ttk.Combobox(sel_frame, values=["Auto Detect"] + LANG_NAMES,
                          width=22, state="readonly")
src_combo.set("Auto Detect")
src_combo.grid(row=0, column=1, padx=8)

tk.Label(sel_frame, text="  →  ", font=("Segoe UI", 14), bg=BG, fg=ACCENT).grid(
    row=0, column=2)

tk.Label(sel_frame, text="Target Language:", font=FONT_LBL, bg=BG, fg=FG).grid(
    row=0, column=3, padx=8)
dst_combo = ttk.Combobox(sel_frame, values=LANG_NAMES, width=22, state="readonly")
dst_combo.set("English")
dst_combo.grid(row=0, column=4, padx=8)

# Input box
tk.Label(root, text="Enter Text:", font=FONT_LBL, bg=BG, fg=FG, anchor="w").pack(
    padx=30, fill="x")
input_box = tk.Text(root, height=6, width=82, font=FONT_TEXT,
                    bg=BOX_BG, fg=FG, insertbackground=FG, relief="flat", padx=6, pady=6)
input_box.pack(padx=30, pady=(2, 8))

# Buttons
btn_frame = tk.Frame(root, bg=BG)
btn_frame.pack()
for txt, cmd, color in [
    ("🔄 Translate", translate_text, "#89b4fa"),
    ("📋 Copy", copy_translation, "#a6e3a1"),
    ("🗑️ Clear", clear_all, "#f38ba8"),
]:
    tk.Button(btn_frame, text=txt, command=cmd, font=FONT_LBL,
              bg=color, fg="#1e1e2e", padx=14, pady=6, relief="flat",
              cursor="hand2").pack(side="left", padx=8)

detected_label = tk.Label(root, text="Detected Language: —",
                           font=("Segoe UI", 10, "italic"), bg=BG, fg="#6c7086")
detected_label.pack(pady=(10, 4))

# Output box
tk.Label(root, text="Translation:", font=FONT_LBL, bg=BG, fg=FG, anchor="w").pack(
    padx=30, fill="x")
output_box = tk.Text(root, height=6, width=82, font=FONT_TEXT,
                     bg=BOX_BG, fg="#a6e3a1", relief="flat", padx=6, pady=6,
                     state="disabled")
output_box.pack(padx=30, pady=(2, 14))

root.mainloop()
