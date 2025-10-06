import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import betas_fitter
import os
from PIL import Image, ImageTk
from tkinter import filedialog

fig = None

def run_script():
    global fig
    try:
        # Récupération des deux vecteurs depuis les champs texte
        x_values = list(map(float, entry_x.get().split(",")))
        y_values = list(map(float, entry_y.get().split(",")))

        # Nettoyer l'ancien tableau
        for row in table.get_children():
            table.delete(row)

        if len(x_values) == len(y_values):
            params, fig, ax = betas_fitter.fitter(x_values,y_values)
        else: 
            messagebox.showerror("Erreur", "Les deux vecteurs doivent avoir la même taille.")
            return
                
        # Remplir le tableau avec les données
        table.insert("", "end", values=(params[0], params[1], params[2], params[3]))

        for widget in plot_frame.winfo_children():
            widget.destroy()  # enlève ancien canvas
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    except ValueError:
        messagebox.showerror("Erreur", "Merci de rentrer uniquement des nombres séparés par des virgules.")

def save_plot(fig):
    if fig is None:
        tk.messagebox.showerror("Erreur", "Aucun plot à sauvegarder !")
        return

    # Ouvrir une boîte de dialogue pour choisir le fichier
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )
    if file_path:
        fig.savefig(file_path)
        tk.messagebox.showinfo("Succès", f"Plot sauvegardé sous {file_path}")

# --- Interface Tkinter ---
root = tk.Tk()
root.title("Beta fitter")
root.geometry("1000x750")

script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "logo.png")
img = Image.open(logo_path)
img = img.resize((180, 100))  # width=50px, height=50px
logo = ImageTk.PhotoImage(img)
label_logo = tk.Label(root, image=logo)
label_logo.pack(pady=3)

# Entrées utilisateur
tk.Label(root, text="Intensitées (séparées par ,)").pack(pady=5)
entry_x = tk.Entry(root, width=50)
entry_x.pack(pady=5)

tk.Label(root, text="Valeurs Cumulées (séparées par ,)").pack(pady=5)
entry_y = tk.Entry(root, width=50)
entry_y.pack(pady=5)

tk.Button(root, text="Générer", command=run_script).pack(pady=10)

# Tableau (Treeview)
columns = ("a", "b", "x min", "x max")
table = ttk.Treeview(root, columns=columns, show="headings", height=2)
for col in columns:
    table.heading(col, text=col)
table.pack(pady=10)

tk.Button(root, text="Exporter le plot en PNG", command=lambda: save_plot(fig=fig)).pack(pady=10)
# Graphique matplotlib intégré
plot_frame = tk.Frame(root)
plot_frame.pack(pady=10, fill="both", expand=True)

root.mainloop()