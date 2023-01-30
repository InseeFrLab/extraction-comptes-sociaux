"""
Constants.
"""

# Liste des colonnes d'intérêts que l'on veut garder dans les tableaux
fields = [
    "Quote-part du capital détenue (en %)",
    "Valeur brute comptable des titres détenus (en K euros)",
    "Valeur nette comptable des titres détenus (en K euros)",
    "Dividendes encaissés par la société en cours d'exercice (en K euros)",
    "Résultats (bénéfices ou pertes à 100%) "
    "du dernier exercice clos (en K devises)",
]

# Liste des pattern regex afin de déterminer les colonnes d'intérêt
regexes = [
    r"\b(quote-part){e<2}\b",
    r"\b(brute){e<2}\b",
    r"\b(nette){e<2}\b",
    r"\b(dividende){e<4}\b",
    r"\b(resultat){e<3}\b",
]
