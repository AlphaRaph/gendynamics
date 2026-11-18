"""Entraine le modèle GaussianFlowDDPM sur un grand nombre d'échantillons d'entrainement différents
Enregistre les résultats dans un fichier csv et crée un graphe pdf dans _figures/_learning_curve"""

import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from gendynamics import GaussianFlowDDPM
from gendynamics.datasets import fetch_synthetic_data
from gendynamics.metrics import sliced_wasserstein
from gendynamics.nn import MLPModel
from gendynamics.training import train
import gc # J'ai ajouté cet import (garbage collector) pour éviter de saturer la mémoire

torch.set_num_threads(2) # J'ai ajouté cette limitation afin que mon ordinateur ne fige pas


def plot(input_file, output_file):
    """Lit les résultats du fichier CSV input_file et crée un graphe pdf avec"""
    n_samples_list = []
    dists = []

    # Lecture du fichier CSV ligne par ligne
    with open(input_file, "r") as f:
        lines = f.readlines()

        # On ignore la première ligne (l'en-tête "n_samples,wasserstein_dist")
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(",")
                if len(parts) == 2:
                    n_samples_list.append(float(parts[0]))
                    dists.append(float(parts[1]))
            except ValueError:
                # Ignore les lignes mal formées si le script a planté en plein milieu d'une écriture
                continue

    # --- Affichage et sauvegarde du graphe ---
    plt.figure(figsize=(8, 5))
    plt.plot(n_samples_list, dists, marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=6)

    plt.xlabel("Nombre d'échantillons d'entraînement ($n_{samples}$)", fontsize=11)
    plt.ylabel("Distance de Wasserstein", fontsize=11)
    plt.title("Évolution de la distance de Wasserstein en fonction de $n_{samples}$", fontsize=12,
              fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)




####################################################################################################
# Main
if __name__ == "__main__":

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="learning_curve.csv",
                        help="CSV file name in _figures/_learning_curve to save the results")
    parser.add_argument("--plot", type=str, default="learning_curve.pdf",
                        help="PDF file name in _figures/_learning_curve to plot the results")
    parser.add_argument("--start", type=int, default=1000, help="First n_samples")
    parser.add_argument("--stop", type=int, default=50000, help="Last n_samples")
    parser.add_argument("--step", type=int, default=1000, help="Step between two n_samples")
    args = parser.parse_args()

    figures_dir = "_figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    learning_curve_dir = "_learning_curve"
    learning_curve_dir = Path(figures_dir / learning_curve_dir)
    learning_curve_dir.mkdir(parents=True, exist_ok=True)

    # --- GESTION DU FICHIER DE SAUVEGARDE ET REPRISE ---
    results_file = learning_curve_dir / args.file

    # Ensemble pour stocker les n_samples déjà traités
    completed_samples = set()

    if results_file.exists():
        print(f"[INFO] Fichier existant détecté : {results_file}. Analyse des données déjà traitées...")
        with open(results_file, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Ignore l'en-tête
                line = line.strip()
                if line:
                    try:
                        n_val = int(float(line.split(",")[0]))
                        completed_samples.add(n_val)
                    except ValueError:
                        continue
        print(f"[INFO] {len(completed_samples)} échantillons déjà calculés trouvés. Ils seront ignorés.")
    else:
        # Si le fichier n'existe pas, on le crée et on écrit l'en-tête
        with open(results_file, "w") as f:
            f.write("n_samples,wasserstein_dist\n")
        print(f"[INFO] Création d'un nouveau fichier de sauvegarde : {results_file}")


    dim = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples_list = np.arange(args.start, args.stop + args.step, args.step)    # Les nombre d'échantillons d'entrainement
    n_test_samples = 5000       # Nombre d'échantillons de test : 100 au lieu de 5000 pour que ce soit plus court
    n_visu_samples = 25         # Nombre de trajectoires affichées
    n_steps = 512
    batch_size = 512    # Le modèle parcourt la base de données d'entrainement par paquet (batch) (pour trouver le
                        # minimum à l'aide d'une descente de gradient) pour aller plus vite (1024 de base)
    n_epochs = 256      # Nombre de fois que le modèle va parcourir l'intégralité de la base de données d'entrainement
                        # (donc par batch) pour finaliser sa descente de gradient
    lr = 5e-4           # learning rate : amplitude des corrections apportées à chaque étape
    width = 128         # Largeur d'une couche du réseau de neurones, je crois, 256 de base
    depth = 4           # Profondeur du réseau de neurones (nombre de couches), je crois, 5 de base
    fdtype = torch.float32
    idtype = torch.int32
    num_workers = 0

    # On choisit le modèle GaussianFlowDDPM
    model = GaussianFlowDDPM

    # On met data à checker comme c'était par défaut (donc un jeu de données en damier)
    data = "checker"

    dists = []

    for n_samples in n_samples_list:
        # --- VÉRIFICATION DE REPRISE ---
        if n_samples in completed_samples:
            print(f"[INFO] n_samples={n_samples} déjà traité. Passage au suivant.")
            continue


        print(f"[INFO] 2D sample test | dataset={data} | device={device} | n_samples={n_samples}"
              f" | n_steps={n_steps} | batch_size={batch_size} | n_epochs={n_epochs}") # J'ai modifié ce print

        # x_ref = target (ce sont les points de la base de données originelle retenus pour les tests)
        x_train, _, x_ref = fetch_synthetic_data(data, n_samples=n_samples, dim=dim, device=device, dtype=fdtype)
        x_ref = x_ref[:n_test_samples]
        train_kwargs = dict(target_data=x_train, batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, num_workers=num_workers)

        print(f"[INFO] Train {model.__name__} | dataset={data}"
              f" | epochs={n_epochs} | batch_size={batch_size} | lr={lr:.1e}")

        print("Creation of the net and the generator...")                   # J'ai ajouté ce print
        net = MLPModel(input_dim=dim, width=width, depth=depth).to(device=device, dtype=fdtype) # Perceptron Multicouches (MLP) : réseau de neurones
        generator = model(net=net, dim=dim, fdtype=fdtype, idtype=idtype, device=device, n_steps=n_steps) # Ce qui va s'occuper d'entrainer le réseau de neurones
        print("Net and generator created.")                                 # J'ai ajouté ce print
        print("Training of the model...")                                   # J'ai ajouté ce print
        train(generative_model=generator, **train_kwargs)
        print("Model trained.")                                             # J'ai ajouté ce print
        print("Generation of the points via the backward process...")       # J'ai ajouté ce print
        x_gen = generator.sample(n_samples=n_test_samples)
        print("Points generated.")                                          # J'ai ajouté ce print

        # On évalue le modèle :
        dist = sliced_wasserstein(x_ref, x_gen)
        print(f"[INFO] Eval {model.__name__} | Wasserstein-dist={dist:.2e}")
        dists.append(dist)

        # --- SAUVEGARDE IMMÉDIATE EN MODE AJOUT (append) ---
        with open(results_file, "a") as f:
            f.write(f"{n_samples},{dist}\n")
        print(f"[INFO] Donnée sauvegardée dans {results_file}")

        del net, generator, x_train, x_ref, x_gen
        gc.collect() # on vide la mémoire

    print(f"[INFO] total={time.time() - t0:.1f}s")
    print(f"[INFO] saved={results_file}")


    plot_file = learning_curve_dir / args.plot

    plot(results_file, plot_file)

    print(f"[INFO] saved={plot_file}")





