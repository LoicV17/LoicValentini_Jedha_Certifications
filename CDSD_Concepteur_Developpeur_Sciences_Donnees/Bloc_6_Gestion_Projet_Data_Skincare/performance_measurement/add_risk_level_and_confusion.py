import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_CSV  = os.path.join(CUR_DIR, "performance_summary.csv")
OUTPUT_CSV = os.path.join(CUR_DIR, "performance_summary_with_risk.csv")
CONF_COUNTS_CSV  = os.path.join(CUR_DIR, "confusion_binary_risk_counts.csv")
CONF_PERCENT_CSV = os.path.join(CUR_DIR, "confusion_binary_risk_percent.csv")
CONF_FIG_PNG     = os.path.join(CUR_DIR, "confusion_binary_risk.png")

# -----------------------------------------------------------------------------
# Constantes / mappings
# -----------------------------------------------------------------------------
MALIGNANT_CLASSES = {"akiec", "bcc", "mel"}
BENIGN_CLASSES    = {"bkl", "df", "nv", "vasc"}

def assign_risk(row):
    """Règles combinées :
       - ÉLEVÉ si (p_bin > 0.50) OU (s_mal > 0.50) OU (classe ∈ {akiec, bcc, mel})
       - FAIBLE si (p_bin < 0.10) ET (s_mal < 0.10) ET (classe ∈ {bkl, df, nv, vasc})
       - Sinon MODÉRÉ
    """
    p_bin_mal = float(row["m1_prob_malignant"])
    s_mal = float(row["m2_prob_malignant_sum"])
    cls = row["m3_pred_class"]

    if (p_bin_mal > 0.50) or (s_mal > 0.50) or (cls in MALIGNANT_CLASSES):
        return "ÉLEVÉ"
    if (p_bin_mal < 0.10) and (s_mal < 0.10) and (cls in BENIGN_CLASSES):
        return "FAIBLE"
    return "MODÉRÉ"

def map_true_binary(true_label: str) -> str:
    """Mappe la vérité terrain multiclasses -> binaire (MALIN / BÉNIN)."""
    return "MALIN" if true_label in MALIGNANT_CLASSES else "BÉNIN"

# -----------------------------------------------------------------------------
# Lecture
# -----------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

required_cols = {"image_id", "true_label", "m1_prob_malignant", "m2_prob_malignant_sum", "m3_pred_class"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Colonnes manquantes dans {INPUT_CSV} : {missing}")

# -----------------------------------------------------------------------------
# Ajout du niveau de risque
# -----------------------------------------------------------------------------
df["risk_level"] = df.apply(assign_risk, axis=1)

# Sauvegarde du CSV enrichi
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Fichier enrichi sauvegardé : {OUTPUT_CSV}")

# -----------------------------------------------------------------------------
# Matrice de confusion (True: MALIN/BÉNIN vs Pred: ÉLEVÉ/MODÉRÉ/FAIBLE)
# -----------------------------------------------------------------------------
# Vérité binaire (à partir de true_label)
df["true_binary"] = df["true_label"].map(map_true_binary)

# Comptes bruts
conf_counts = pd.crosstab(
    df["true_binary"],
    df["risk_level"],
    rownames=["Vrai"],
    colnames=["Prédit"],
    dropna=False
).reindex(index=["MALIN", "BÉNIN"], columns=["ÉLEVÉ", "MODÉRÉ", "FAIBLE"], fill_value=0)

conf_counts.to_csv(CONF_COUNTS_CSV)
print(f"💾 Confusion (comptes) sauvegardée : {CONF_COUNTS_CSV}")

# Pourcentages par ligne (normalisée true)
conf_percent = conf_counts.div(conf_counts.sum(axis=1).replace(0, 1), axis=0) * 100.0
conf_percent.to_csv(CONF_PERCENT_CSV)
print(f"💾 Confusion (pourcentages) sauvegardée : {CONF_PERCENT_CSV}")

# Heatmap (%)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_percent, annot=True, fmt=".1f", cmap="Blues",
            vmin=0, vmax=100, cbar_kws={"label": "% par classe vraie"})
plt.title("Matrice de confusion (%) — Vrai: Malin/Bénin vs Prédit: Élevé/Modéré/Faible")
plt.ylabel("Vrai")
plt.xlabel("Prédit")
plt.tight_layout()
plt.savefig(CONF_FIG_PNG, dpi=150)
plt.close()
print(f"🖼️ Heatmap sauvegardée : {CONF_FIG_PNG}")

# -----------------------------------------------------------------------------
# Résumé utile en console
# -----------------------------------------------------------------------------
print("\nRésumé distribution des niveaux de risque prédits :")
print(df["risk_level"].value_counts().to_string())

print("\nAperçu (5 premières lignes) :")
print(df[["image_id", "true_label", "true_binary", "m1_prob_malignant",
          "m2_prob_malignant_sum", "m3_pred_class", "risk_level"]].head())
