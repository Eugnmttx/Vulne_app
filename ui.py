import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import betas_fitter
import os
from PIL import Image, ImageTk
import csv
import sys
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg
import webbrowser
import subprocess

# Variables globales
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
        title="Import CSV file (per rows or columns)",
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
            messagebox.showerror("Error", "The CSV file is empty.")
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
            messagebox.showerror("Error", "No valid dataset found.")
            return

        dataset_names = list(datasets.keys())
        current_index = 0
        show_current_dataset()
        messagebox.showinfo(
            "Importation successful",
            f"{len(datasets)} datasets detected ({'columns' if is_column_oriented else 'rows'})."
        )

    except Exception as e:
        messagebox.showerror("Error", f"Impossible to read CSV :\n{e}")

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

# -- Enlever une ligne manuellement --
def delete_row():
    global datasets, dataset_names, current_index

    if not dataset_names:
        messagebox.showerror("Error", "No dataset loaded.")
        return

    name = dataset_names[current_index]
    x_vals, y_vals = datasets[name]

    selected = data_table.selection()
    if not selected:
        messagebox.showwarning("Warning", "No row selected.")
        return

    item_id = selected[0]
    row_index = data_table.index(item_id)

    # Remove from Treeview
    data_table.delete(item_id)

    # Remove from internal data
    if row_index < len(x_vals):
        x_vals.pop(row_index)
    if row_index < len(y_vals):
        y_vals.pop(row_index)

    datasets[name] = (x_vals, y_vals)

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
        messagebox.showerror("Error", "No dataset imported.")
        return
    name = dataset_names[current_index]
    x_values, y_values = datasets[name]

    try:
        result = betas_fitter.fitter(
            x_values, y_values, name,
            x_label=x_label_var.get(),
            y_label=y_label_var.get(),
            plot_error=plot_error.get(),
            log_scale=log_scale.get(),
            minmax=minmax.get()
        )

        if isinstance(result, tuple) and len(result) == 5:
            params, params_max, params_min, fig, ax = result
            param_table.delete(*param_table.get_children())
            param_table.insert("", "end", values=("fit", *params))
            param_table.insert("", "end", values=("fit max", *params_max))
            param_table.insert("", "end", values=("fit min", *params_min))
        elif isinstance(result, tuple) and len(result) == 3:
            params, fig, ax = result
            param_table.delete(*param_table.get_children())
            param_table.insert("", "end", values=("fit", *params))
        else:
            raise ValueError(f"Unexpected return from fitter: got {type(result)} len={len(result) if isinstance(result,tuple) else 'N/A'}")

        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", f"Generating error : {e}")

# --- Sauvegarde du plot ---
def save_plot(fig):
    if fig is None:
        messagebox.showerror("Error", "No plot to save !")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )
    if file_path:
        fig.set_size_inches(8, 6)
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        messagebox.showinfo("Success", f"Plot saved at {file_path}")

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
    messagebox.showinfo("Success", "Parameters copied in the clipboard !")

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
                messagebox.showerror("Error", f"Non numerical value : {new_val}")
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
            messagebox.showerror("Errorr", f"Non numerical value : {new_val}")
            return
        param_table.set(item_id, column, new_val)

        if col_index == 0:
            entry.destroy()
            return
        
        param_idx = col_index - 1
        if 0 <= param_idx < len(current_params):
            current_params[param_idx] = fval
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
    for col_index, col in enumerate(param_columns[1:]):
        val = param_table.set(item_id, col)
        try:
            current_params[col_index] = float(val.replace(',', '.'))
        except ValueError:
            current_params[col_index] = 0  # fallback si valeur invalide
            
# -- Live replotting of the fig -- 
def replot_manual():
    global fig
    if not dataset_names:
        messagebox.showerror("Error", "No data was imported.")
        return

    save_param_edits()  # <-- force la lecture de toutes les cellules éditées

    name = dataset_names[current_index]
    x_values, y_values = datasets[name]

    try:
        fig = betas_fitter.manual_fig(current_params, x_values, y_values, name, x_label=x_label_var.get(), y_label=y_label_var.get(), plot_error=plot_error.get(), log_scale=log_scale.get(), minmax=minmax.get())
        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", f"Error while plotting : {e}")

# --- Interface Tkinter ---
root = tk.Tk()
root.title("Beta fitter")

# Fenêtre redimensionnable
root.geometry("1000x750")
root.minsize(900, 650)

# --- Canvas + Scrollbar verticale ---
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(main_frame, highlightthickness=0)
canvas.pack(side="left", fill="both", expand=True)

scrollbar_y = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollbar_y.pack(side="right", fill="y")

canvas.configure(yscrollcommand=scrollbar_y.set)

# --- Frame interne (interface principale) ---
scrollable_frame = tk.Frame(canvas)
# Création correcte de la fenêtre dans le canvas
canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="n")

# Ajuste la zone scrollable quand le contenu change
def update_scrollregion(event=None):
    canvas.configure(scrollregion=canvas.bbox("all"))
    # Centre horizontalement si la fenêtre est plus large
    canvas.itemconfig(canvas_window, width=canvas.winfo_width())

scrollable_frame.bind("<Configure>", update_scrollregion)

# -- Fonction de scroll
def _on_mousewheel(event):
    """
    Cross-platform mouse wheel handler.
    - Windows: event.delta is multiple of 120
    - macOS:   event.delta is small (1/-1)
    - Linux:   use Button-4 (up) / Button-5 (down)
    """
    # Linux (X11) uses Button-4/5 (no event.delta)
    if hasattr(event, "num") and event.num in (4, 5):
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
        return

    # Otherwise we have event.delta (Windows / macOS)
    delta = event.delta
    if sys.platform == "darwin":
        # On macOS delta is typically small (1 or -1) — use directly
        canvas.yview_scroll(int(-1 * delta), "units")
    else:
        # On Windows delta is multiple of 120
        canvas.yview_scroll(int(-1 * (delta / 120)), "units")

# Bind globally so the scrolling works whatever has the focus
root.bind_all("<MouseWheel>", _on_mousewheel, add="+")   # Windows / macOS
root.bind_all("<Button-4>", _on_mousewheel, add="+")     # Linux scroll up
root.bind_all("<Button-5>", _on_mousewheel, add="+")     # Linux scroll down

# --------------------------------------------------------
#                CONTENU DE L'INTERFACE
# --------------------------------------------------------

# Logo
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "logo.png")
if os.path.exists(logo_path):
    img = Image.open(logo_path).resize((180, 100))
    logo = ImageTk.PhotoImage(img)
    tk.Label(scrollable_frame, image=logo).pack(pady=3)

# Bouton import
tk.Button(scrollable_frame, text="📂 Import CSV", command=import_csv_multi).pack(pady=5)

# Navigation dataset
nav_frame = tk.Frame(scrollable_frame)
nav_frame.pack(pady=5)
tk.Button(nav_frame, text="◀", command=prev_dataset, width=4).pack(side="left", padx=5)
dataset_label = tk.Label(nav_frame, text="No dataset")
dataset_label.pack(side="left", padx=10)
tk.Button(nav_frame, text="▶", command=next_dataset, width=4).pack(side="left", padx=5)

# Tableau de données
columns = ("x", "y")
data_table = ttk.Treeview(scrollable_frame, columns=columns, show="headings", height=10)
for col in columns:
    data_table.heading(col, text=col)
    data_table.column(col, width=120)
data_table.pack(pady=5)
data_table.bind("<Double-1>", lambda e: edit_cell(data_table, e))

data_management_frame = tk.Frame(scrollable_frame)
data_management_frame.pack(pady=5)
tk.Button(data_management_frame, text="+", font=('Helvetica',14,"bold"), command=add_row).pack(side='left', padx=5)
tk.Button(data_management_frame, text="-", font=('Helvetica',14,"bold"), command=delete_row).pack(side='left', padx=5)

# Label of axes
axis_container = tk.Frame(scrollable_frame)
axis_container.pack(pady=10, fill='x')

axis_frame = tk.LabelFrame(axis_container, text="Axes labels", padx=10, pady=5)
axis_frame.pack(anchor="center")  # <-- Center the frame horizontally

tk.Label(axis_frame, text="X-axis label:").grid(row=0, column=0, sticky="e", padx=5, pady=3)
x_label_var = tk.StringVar(value=None)
x_label_entry = tk.Entry(axis_frame, textvariable=x_label_var, width=30, justify="center")
x_label_entry.grid(row=0, column=1, padx=5, pady=3)
log_scale = tk.BooleanVar(value=False)
tk.Checkbutton(axis_frame, text="Log scale", variable=log_scale).grid(row=0, column=4, sticky="e", padx=5, pady=3)

tk.Label(axis_frame, text="Y-axis label:").grid(row=1, column=0, sticky="e", padx=5, pady=3)
y_label_var = tk.StringVar(value=None)
y_label_entry = tk.Entry(axis_frame, textvariable=y_label_var, width=30, justify="center")
y_label_entry.grid(row=1, column=1, padx=5, pady=3)

# Bouton générer
generate_frame = tk.Frame(scrollable_frame)
generate_frame.pack(pady=5)
plot_error = tk.BooleanVar(value=True)
minmax = tk.BooleanVar(value=False)
tk.Checkbutton(generate_frame, text="Plot with error", variable=plot_error).pack(side="left", padx=5)
tk.Checkbutton(generate_frame, text="Plot with Min-Max", variable=minmax).pack(side="left", padx=5)
tk.Button(generate_frame, text="Generate plot", command=run_script).pack(side="right", padx=5)

# Essai de tableau affichable
test_frame = tk.Frame(scrollable_frame)
test_frame.pack(pady=5)
sub_frame = tk.Frame(test_frame)
sub_table = ttk.Treeview(sub_frame, columns='test', show='headings')
sub_table.pack(pady=5)

def toggle_minmax(*args):
    if minmax.get():
        test_frame.pack(pady=5, fill='x', before=param_table)
        sub_frame.pack(pady=5, fill="x")
    else:
        sub_frame.pack_forget()
        test_frame.pack_forget()
    # force la mise à jour de la zone scrollable pour que le widget apparaisse/disparaisse
    try:
        update_scrollregion()
    except Exception:
        canvas.update_idletasks()
#minmax.trace_add("write", toggle_minmax)


# Tableau des paramètres
param_columns = ("","a", "b", "x min", "x max")
param_table = ttk.Treeview(scrollable_frame, columns=param_columns, show="headings", height=3)
for col in param_columns:
    param_table.heading(col, text=col)
    w = 50 if col == "" else 110
    param_table.column(col, width=w, anchor="center", stretch=True)
param_table.pack(pady=5)
param_table.bind("<Double-1>", edit_param_cell)

btn_param_frame = tk.Frame(scrollable_frame)
btn_param_frame.pack(pady=5)
tk.Button(btn_param_frame, text="Plot with modified parameters", command=replot_manual).pack(side="left",padx=5)

# Boutons copier/export
btn_frame = tk.Frame(scrollable_frame)
btn_frame.pack(pady=5)
tk.Button(btn_frame, text="Copy parameters", command=copy_params_to_clipboard).pack(side="left", padx=5)
tk.Button(btn_frame, text="Export plot", command=lambda: save_plot(fig)).pack(side="left", padx=5)

# Zone plot
plot_frame = tk.Frame(scrollable_frame)
plot_frame.pack(pady=10, fill="both", expand=True)

# -- Help window --
def open_help_window():
    """Ouvre une fenêtre d'aide avec du texte et des formules LaTeX."""
    help_win = tk.Toplevel(root)
    help_win.title("Guide - Beta fitter")
    help_win.geometry("800x300")

    # --- Scrollable Frame pour le texte ---
    canvas = tk.Canvas(help_win)
    scrollbar = ttk.Scrollbar(help_win, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # --- Contenu textuel ---
    intro = (
        "Welcome in Beta fitter\n"
        "This app fits experimental data with a cumulative beta distribution function.\n"
        "Here are a few details for usage:\n\n"
        "(1) For importation, the file can be either sorted by row or by column. Each row (column) must look like: ['header', I_1, ..., I_N, V_1, ..., V_N] where I_i, V_i are respectively the intensities and the vulnerabilities.\n\n"
        "(2) You can naviguate your data set with the arrows, and manually modify the content of the data by double-clicking on a cell.\n\n"
        "(3) Once you've plotted your graph, you can modify the parameters of the fit and re-generate the plot to see how the distribution reacts.\n\n"

        "For further details about the fitting algorithm or the mathematical context click here: "
    )

    tk.Label(scroll_frame, text=intro, justify="left", wraplength=760).pack(pady=10, padx=10)
    
    def open_help_pdf():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(script_dir, "details.pdf")

        if not os.path.exists(pdf_path):
            messagebox.showerror("Erreur", f"Le fichier PDF {pdf_path} est introuvable.")
            return

        try:
            if sys.platform.startswith("darwin"):  # macOS
                subprocess.call(["open", pdf_path])
            elif os.name == "nt":  # Windows
                os.startfile(pdf_path)
            elif os.name == "posix":  # Linux and others
                subprocess.call(["xdg-open", pdf_path])
            else:
                # Fallback to webbrowser if none of the above works
                import webbrowser
                webbrowser.open_new(pdf_path)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir le fichier PDF : {e}")

    # --- Button in main window ---
    tk.Button(scroll_frame, text="Mathematical details", command=open_help_pdf).pack(pady=5)

# --- Bouton pour ouvrir le guide ---
tk.Button(root, text="Help", command=open_help_window).pack(pady=5)
# --------------------------------------------------------
root.mainloop()

if __name__ == "__main__":
    # Protection pour les exécutables Windows (PyInstaller)
    import multiprocessing
    multiprocessing.freeze_support()