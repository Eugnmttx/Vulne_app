import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import betas_fitter
import os
from PIL import Image, ImageTk
import csv

fig = None
datasets = {}   # name -> (x_values, y_values)
dataset_names = []
current_index = 0

# --- Fonction utilitaire ---
def is_number(v):
    try:
        float(v.replace(",", "."))
        return True
    except ValueError:
        return False

# --- Import CSV avec détection automatique ---
def import_csv_multi():
    """Import CSV et détecte automatiquement si format par lignes ou par colonnes."""
    global datasets, dataset_names, current_index
    file_path = filedialog.askopenfilename(
        title="Importer un fichier CSV (multi-datasets en lignes ou colonnes)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not file_path:
        return

    try:
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            sample = csvfile.read(1024)
            csvfile.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
            reader = csv.reader(csvfile, dialect)
            rows = [list(map(str.strip, r)) for r in reader if any(r)]

        if not rows:
            messagebox.showerror("Erreur", "Le fichier CSV est vide.")
            return

        # --- Détection du format ---
        first_row = rows[0]
        num_non_numeric = sum(not is_number(v) for v in first_row)
        is_column_oriented = num_non_numeric >= 2  # plusieurs non-nombres = header → colonnes

        datasets.clear()

        if not is_column_oriented:
            # 🟩 Format "une ligne = dataset"
            for i, row in enumerate(rows):
                if len(row) < 2:
                    continue
                # Nom du dataset = première cellule, fallback si vide
                name = row[0] if row[0] else f"dataset_{i+1}"
                numeric_values = [float(v.replace(",", ".")) for v in row[1:] if is_number(v)]
                if len(numeric_values) >= 2:
                    half = len(numeric_values) // 2
                    x_vals = numeric_values[:half]
                    y_vals = numeric_values[half:]
                    datasets[name] = (x_vals, y_vals)
        else:
            # 🟦 Format "une colonne = dataset"
            header = rows[0]
            data_rows = rows[1:]
            num_rows = len(data_rows)
            num_cols = len(header)

            for j in range(num_cols):
                name = header[j] if header[j] else f"dataset_{j+1}"
                # Récupère les valeurs de la colonne j
                col_vals = [data_rows[i][j] for i in range(num_rows) if j < len(data_rows[i])]
                numeric_values = [float(v.replace(",", ".")) for v in col_vals if is_number(v)]
                if len(numeric_values) >= 2:
                    half = len(numeric_values) // 2
                    x_vals = numeric_values[:half]
                    y_vals = numeric_values[half:]
                    datasets[name] = (x_vals, y_vals)

        if not datasets:
            messagebox.showerror("Erreur", "Aucun dataset valide trouvé.")
            return

        dataset_names = list(datasets.keys())
        current_index = 0
        show_current_dataset()
        messagebox.showinfo(
            "Importation réussie",
            f"{len(datasets)} jeux de données détectés ({'colonnes' if is_column_oriented else 'lignes'})."
        )

    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de lire le CSV :\n{e}")

# -- Ajouter une ligne manuellement --
def add_row():
    global datasets, dataset_names, current_index

    if not dataset_names:
        # No dataset yet → create a default one
        default_name = "dataset_1"
        datasets[default_name] = ([], [])
        dataset_names.append(default_name)
        current_index = 0
        show_current_dataset()

    # Add empty row to Treeview
    data_table.insert("", "end", values=("", ""))

# --- Affichage d'un dataset dans le tableau ---
def show_current_dataset():
    if not dataset_names:
        return
    name = dataset_names[current_index]
    x_vals, y_vals = datasets[name]

    # Effacer ancien contenu
    data_table.delete(*data_table.get_children())
    # Ajouter les valeurs
    for x, y in zip(x_vals, y_vals):
        data_table.insert("", "end", values=(x, y))
    dataset_label.config(text=f"Dataset : {name} ({current_index+1}/{len(dataset_names)})")

# --- Navigation ---
def prev_dataset():
    global current_index
    if not dataset_names:
        return
    current_index = (current_index - 1) % len(dataset_names)
    show_current_dataset()

def next_dataset():
    global current_index
    if not dataset_names:
        return
    current_index = (current_index + 1) % len(dataset_names)
    show_current_dataset()

# --- Génération du plot ---
def run_script():
    global fig
    if not dataset_names:
        messagebox.showerror("Erreur", "Aucun dataset importé.")
        return
    name = dataset_names[current_index]
    x_values, y_values = datasets[name]

    try:
        params, fig, ax = betass_fitter.fitter(x_values, y_values, name)
        param_table.delete(*param_table.get_children())
        param_table.insert("", "end", values=(params[0], params[1], params[2], params[3]))

        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Erreur", f"Erreur de génération : {e}")

# --- Sauvegarde du plot ---
def save_plot(fig):
    if fig is None:
        messagebox.showerror("Erreur", "Aucun plot à sauvegarder !")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )
    if file_path:
        fig.set_size_inches(8, 6)
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        messagebox.showinfo("Succès", f"Plot sauvegardé sous {file_path}")

# --- Copier les paramètres ---
def copy_params_to_clipboard():
    all_rows = []
    for child in param_table.get_children():
        values = param_table.item(child)["values"]
        all_rows.append("\t".join(map(str, values)))
    clipboard_text = "\n".join(all_rows)
    root.clipboard_clear()
    root.clipboard_append(clipboard_text)
    root.update()
    messagebox.showinfo("Succès", "Paramètres copiés dans le presse-papiers !")

# -- Modifier les cellules manuellement --
def edit_cell(tree, event):
    global datasets, dataset_names, current_index

    item_id = tree.identify_row(event.y)
    column = tree.identify_column(event.x)
    if not item_id or not column:
        return

    col_index = int(column.replace('#','')) - 1
    x, y, width, height = tree.bbox(item_id, column)
    value = tree.set(item_id, column)

    entry = tk.Entry(tree)
    entry.place(x=x, y=y, width=width, height=height)
    entry.insert(0, value)
    entry.focus()

    def save_edit(event=None):
        new_val = entry.get()
        tree.set(item_id, column, new_val)
        entry.destroy()

        if dataset_names and 0 <= current_index < len(dataset_names):
            name = dataset_names[current_index]
            x_vals, y_vals = datasets[name]

            row_index = tree.index(item_id)
            try:
                new_val_float = float(new_val.replace(',', '.'))
            except ValueError:
                messagebox.showerror("Erreur", f"Valeur non numérique : {new_val}")
                return

            # If editing a new row, append to the list
            if col_index == 0:
                if row_index < len(x_vals):
                    x_vals[row_index] = new_val_float
                else:
                    # Extend x_vals with new element
                    while len(x_vals) < row_index:
                        x_vals.append(0.0)
                    x_vals.append(new_val_float)
                    # Make sure y_vals matches length
                    while len(y_vals) < len(x_vals):
                        y_vals.append(0.0)
            elif col_index == 1:
                if row_index < len(y_vals):
                    y_vals[row_index] = new_val_float
                else:
                    # Extend y_vals with new element
                    while len(y_vals) < row_index:
                        y_vals.append(0.0)
                    y_vals.append(new_val_float)
                    # Make sure x_vals matches length
                    while len(x_vals) < len(y_vals):
                        x_vals.append(0.0)

            datasets[name] = (x_vals, y_vals)

    entry.bind("<Return>", save_edit)
    entry.bind("<FocusOut>", save_edit)

# -- Edit parameters cell --
current_params = [0, 0, 0, 0]  # store the last fitted parameters

def edit_param_cell(event):
    """Edit param_table cells and update current_params"""
    item_id = param_table.identify_row(event.y)
    column = param_table.identify_column(event.x)
    if not item_id or not column:
        return

    col_index = int(column.replace('#','')) - 1
    x, y, width, height = param_table.bbox(item_id, column)
    value = param_table.set(item_id, column)

    entry = tk.Entry(param_table)
    entry.place(x=x, y=y, width=width, height=height)
    entry.insert(0, value)
    entry.focus()

    def save_edit(event=None):
        new_val = entry.get()
        try:
            fval = float(new_val.replace(',', '.'))
        except ValueError:
            messagebox.showerror("Erreur", f"Valeur non numérique : {new_val}")
            return
        param_table.set(item_id, column, new_val)
        current_params[col_index] = fval  # update current_params
        entry.destroy()

    entry.bind("<Return>", save_edit)
    entry.bind("<FocusOut>", save_edit)

# -- On sauvegarde avant de replot --
def save_param_edits():
    """Force l'application de toutes les valeurs présentes dans le param_table"""
    global current_params
    children = param_table.get_children()
    if not children:
        return
    item_id = children[0]  # on n'a qu'une ligne dans param_table
    for col_index, col in enumerate(param_columns):
        val = param_table.set(item_id, col)
        try:
            current_params[col_index] = float(val.replace(',', '.'))
        except ValueError:
            current_params[col_index] = 0  # fallback si valeur invalide
# -- Live replotting of the fig -- 
def replot_manual():
    global fig
    if not dataset_names:
        messagebox.showerror("Erreur", "Aucun dataset importé.")
        return

    save_param_edits()  # <-- force la lecture de toutes les cellules éditées

    name = dataset_names[current_index]
    x_values, y_values = datasets[name]

    try:
        fig = betas_fitter.manual_fig(current_params, x_values, y_values, name)
        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Erreur", f"Erreur lors du replot : {e}")

# --- Interface Tkinter ---
root = tk.Tk()
root.title("Beta fitter - Multi dataset viewer")
root.geometry("1000x750")

# Logo
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "logo.png")
if os.path.exists(logo_path):
    img = Image.open(logo_path).resize((180, 100))
    logo = ImageTk.PhotoImage(img)
    tk.Label(root, image=logo).pack(pady=3)

# Bouton import
tk.Button(root, text="📂 Importer CSV", command=import_csv_multi).pack(pady=5)

# Zone dataset
nav_frame = tk.Frame(root)
nav_frame.pack(pady=5)
tk.Button(nav_frame, text="◀", command=prev_dataset, width=4).pack(side="left", padx=5)
dataset_label = tk.Label(nav_frame, text="Aucun dataset")
dataset_label.pack(side="left", padx=10)
tk.Button(nav_frame, text="▶", command=next_dataset, width=4).pack(side="left", padx=5)
tk.Button(root, text="Ajouter ligne", command=add_row).pack(pady=5)

# Tableau de données
columns = ("x", "y")
data_table = ttk.Treeview(root, columns=columns, show="headings", height=10)
for col in columns:
    data_table.heading(col, text=col)
    data_table.column(col, width=120)
data_table.pack(pady=5)
data_table.bind("<Double-1>", lambda e: edit_cell(data_table, e))

# Bouton générer
tk.Button(root, text="Générer le plot", command=run_script).pack(pady=10)

# Tableau paramètres
param_columns = ("a", "b", "x min", "x max")
param_table = ttk.Treeview(root, columns=param_columns, show="headings", height=2)
for col in param_columns:
    param_table.heading(col, text=col)
param_table.pack(pady=5)
param_table.bind("<Double-1>", edit_param_cell)
tk.Button(root, text="Replot avec paramètres modifiés", command=replot_manual).pack(pady=5)

# Boutons copier/export
btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)
tk.Button(btn_frame, text="Copier paramètres", command=copy_params_to_clipboard).pack(side="left", padx=5)
tk.Button(btn_frame, text="Exporter le plot", command=lambda: save_plot(fig)).pack(side="left", padx=5)

# Zone plot
plot_frame = tk.Frame(root)
plot_frame.pack(pady=10, fill="both", expand=True)

root.mainloop()
