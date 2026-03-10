"""import numpy as np

class Filtre_ANC:
    def __init__(self, mu, ordre_du_filtre): # On ajoute mu ici
        self.mu = mu
        self.poids = np.zeros(ordre_du_filtre)
        self.memoire = np.zeros(ordre_du_filtre)

    def calculer_anti_bruit(self, echantillon_entree, echantillon_erreur):
        # 1. Mise à jour de la mémoire (ligne à retard)
        self.memoire = np.roll(self.memoire, 1)
        self.memoire[0] = echantillon_entree

        # 2. Calcul du signal de sortie (Anti-Bruit)
        # y(n) = W . X
        anti_bruit = np.dot(self.poids, self.memoire)

        # 3. Mise à jour des poids (Algorithme LMS)
        # W(n+1) = W(n) + mu * e(n) * X(n)
        self.poids += self.mu * echantillon_erreur * self.memoire

        return anti_bruit

# --- EXECUTION ---
# On crée le filtre (mu=0.01, ordre=16)
filtre = Filtre_ANC(0.01, 16)

# Simulation sur une boucle (pour traiter tout ton signal_entree)
resultats_antibruit = []

for i in range(len(signal_entree)):
    # Note : echantillon_erreur[i] doit provenir de tes données Latis-Pro
    y = filtre.calculer_anti_bruit(signal_entree[i], signal_erreur[i])
    resultats_antibruit.append(y)


# Après avoir calculé ton tableau resultats_antibruit
signal_somme = np.array(signal_entree) + np.array(resultats_antibruit)

# Affichage des 3 courbes
import matplotlib.pyplot as plt

plt.plot(signal_entree, label="Bruit initial (Micro ref)")
plt.plot(resultats_antibruit, label="Anti-bruit généré")
plt.plot(signal_somme, label="Résidu (Somme)", linewidth=2, color='red')
plt.legend()
plt.show()
"""
import numpy as np
import sounddevice as sd

# --- Paramètres de l'expérience ---
FS = 16000          # Fréquence d'échantillonnage (16kHz suffisent pour du 500Hz)
BLOCK_SIZE = 256    # Taille du buffer (plus c'est petit, moins il y a de latence)
MU = 0.01           # Pas d'adaptation (vitesse d'apprentissage de l'algorithme)
N_TAPS = 32         # Nombre de coefficients du filtre (longueur du filtre)

# Initialisation du filtre
w = np.zeros(N_TAPS)
x_buffer = np.zeros(N_TAPS)

def callback(indata, outdata, frames, time, status):
    global w, x_buffer
    
    if status:
        print(status)

    # Séparation des canaux des micros INMP441
    # Canal 0 (GND sur SEL) : Micro de Référence (le bruit extérieur)
    # Canal 1 (3.3V sur SEL) : Micro d'Erreur (le résultat après annulation)
    ref_signal = indata[:, 0]
    err_signal = indata[:, 1]
    
    output = np.zeros(frames)

    for i in range(frames):
        # 1. Mise à jour du buffer d'entrée
        x_buffer = np.roll(x_buffer, 1)
        x_buffer[0] = ref_signal[i]
        
        # 2. Calcul de l'anti-bruit (Produit scalaire)
        y = np.dot(w, x_buffer)
        output[i] = -y # Opposition de phase
        
        # 3. Mise à jour des coefficients du filtre (LMS)
        # On ajuste w pour minimiser l'erreur captée par le 2ème micro
        w = w + 2 * MU * err_signal[i] * x_buffer

    # Envoi de l'anti-bruit vers l'amplificateur I2S
    outdata[:, 0] = output  # Sortie mono sur le canal gauche
    outdata[:, 1] = output  # Copie sur le canal droit (si besoin)

# --- Lancement du flux temps réel ---
try:
    with sd.Stream(channels=2, callback=callback, samplerate=FS, blocksize=BLOCK_SIZE):
        print("Réduction active en cours... Appuyez sur Ctrl+C pour arrêter.")
        while True:
            pass
except KeyboardInterrupt:
    print("\nArrêt du système.")
