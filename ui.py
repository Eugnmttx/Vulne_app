import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import betas_fitter
import os
from PIL import Image, ImageTk
import csv

fig = None

# --- Script functions ---
def run_script():
    global fig
    try:
        x_values = []
        y_values = []
        for child in table.get_children():
            x, y = table.item(child)["values"]
            x_values.append(float(x))
            y_values.append(float(y))

        if len(x_values) == len(y_values) and len(x_values) > 0:
            params, fig, ax = betas_fitter.fitter(x_values, y_values)
        else:
            messagebox.showerror("Erreur", "Les colonnes doivent avoir la même taille et ne pas être vides.")
            return

        # Afficher les paramètres
        param_table.delete(*param_table.get_children())
        param_table.insert("", "end", values=(params[0], params[1], params[2], params[3]))

        # Afficher le plot
        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    except ValueError:
        messagebox.showerror("Erreur", "Merci de rentrer uniquement des nombres valides.")

def save_plot(fig):
    if fig is None:
        tk.messagebox.showerror("Erreur", "Aucun plot à sauvegarder !")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )
    if file_path:
        # Standard size and high resolution
        fig.set_size_inches(8, 6)  # largeur x hauteur en pouces
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        tk.messagebox.showinfo("Succès", f"Plot sauvegardé sous {file_path}")

# --- Edit in-place for Treeview ---
def on_double_click(event):
    item = table.identify_row(event.y)
    column = table.identify_column(event.x)
    if not item or not column:
        return
    x, y, width, height = table.bbox(item, column)
    value = table.set(item, column)

    entry = tk.Entry(table)
    entry.place(x=x, y=y, width=width, height=height)
    entry.insert(0, value)
    entry.focus()

    def save_edit(event=None):
        table.set(item, column, entry.get())
        entry.destroy()

    entry.bind("<Return>", save_edit)
    entry.bind("<FocusOut>", save_edit)

# --- Paste from Excel ---
def paste_from_clipboard(event=None):
    try:
        clipboard = root.clipboard_get()
        rows = clipboard.strip().split("\n")
        table.delete(*table.get_children())  # clear existing rows
        for row in rows:
            values = row.split("\t")  # Excel usually copies tab-delimited
            if len(values) >= 2:
                table.insert("", "end", values=(values[0], values[1]))
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de coller depuis le presse-papiers.\n{e}")

# --- Add/Remove rows ---
def add_row():
    table.insert("", "end", values=("", ""))

def remove_selected_row():
    selected = table.selection()
    for item in selected:
        table.delete(item)

# -- Get params to clipboard ---
def copy_params_to_clipboard():
    all_rows = []
    for child in param_table.get_children():
        values = param_table.item(child)["values"]
        all_rows.append("\t".join(map(str, values)))
    clipboard_text = "\n".join(all_rows)
    root.clipboard_clear()
    root.clipboard_append(clipboard_text)
    root.update()  # assure que le clipboard est mis à jour
    messagebox.showinfo("Succès", "Paramètres copiés dans le presse-papiers !")

# -- Import csv in table --
def import_csv():
    file_path = filedialog.askopenfilename(
        title="Importer un fichier CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not file_path:
        return

    try:
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            # Détection automatique du séparateur
            sample = csvfile.read(1024)
            csvfile.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
            reader = csv.reader(csvfile, dialect)
            rows = list(reader)

        if not rows:
            messagebox.showerror("Erreur", "Le fichier CSV est vide.")
            return

        # Nettoyer la table existante
        table.delete(*table.get_children())

        def is_number(value):
            """Vérifie si une valeur est numérique après conversion de la virgule en point."""
            try:
                float(value.replace(",", "."))
                return True
            except ValueError:
                return False

        # Déterminer si la première ligne est un en-tête
        first_row = rows[0]
        start_index = 1 if not all(is_number(v) for v in first_row[:2]) else 0

        # Importer les deux premières colonnes en remplaçant les virgules par des points
        for row in rows[start_index:]:
            if len(row) >= 2:
                x = row[0].strip().replace(",", ".")
                y = row[1].strip().replace(",", ".")
                table.insert("", "end", values=(x, y))

        messagebox.showinfo("Succès", f"Importation réussie depuis {os.path.basename(file_path)}.")

    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de lire le fichier CSV :\n{e}")

# --- Tkinter GUI ---
root = tk.Tk()
root.title("Beta fitter")
root.geometry("1000x750")

# Logo
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "logo.png")
img = Image.open(logo_path)
img = img.resize((180, 100))
logo = ImageTk.PhotoImage(img)
label_logo = tk.Label(root, image=logo)
label_logo.pack(pady=3)

# Input table
tk.Label(root, text="Entrez vos valeurs d'intensité et de vulnerabilité (double-click pour éditer ou Ctrl+V pour coller depuis Excel)").pack(pady=5)
columns = ("Intensité", "vulnérabilité")
table = ttk.Treeview(root, columns=columns, show="headings", height=10)
for col in columns:
    table.heading(col, text=col)
    table.column(col, width=100)

# Start with 2 empty rows
for _ in range(2):
    table.insert("", "end", values=("", ""))

table.pack(pady=5, fill="x")
table.bind("<Double-1>", on_double_click)
root.bind_all("<Control-v>", paste_from_clipboard)  # Ctrl+V anywhere pastes into table

# Add/remove row buttons
row_frame = tk.Frame(root)
row_frame.pack(pady=5)

tk.Button(row_frame, text="Importer CSV", command=import_csv).pack(side="left", padx=5)
tk.Button(row_frame, text="Ajouter ligne", command=add_row).pack(side="left", padx=5)
tk.Button(row_frame, text="Supprimer ligne(s)", command=remove_selected_row).pack(side="left", padx=5)
tk.Button(row_frame, text="Supprimer tout", command=lambda: table.delete(*table.get_children())).pack(side="left", padx=5)

# Generate button
tk.Button(root, text="Générer", command=run_script).pack(pady=10)

# Parameter table
param_columns = ("a", "b", "x min", "x max")
param_table = ttk.Treeview(root, columns=param_columns, show="headings", height=2)
for col in param_columns:
    param_table.heading(col, text=col)
param_table.pack(pady=5)

buttons_frame = tk.Frame(root)
buttons_frame.pack(pady=5)
tk.Button(root, text="Copier les paramètres", command=copy_params_to_clipboard).pack(padx=5)
tk.Button(root, text="Exporter le plot en PNG", command=lambda: save_plot(fig=fig)).pack(padx=5)

# Plot frame
plot_frame = tk.Frame(root)
plot_frame.pack(pady=10, fill="both", expand=True)

root.mainloop()