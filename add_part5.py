import json

with open(r'c:\indusense\Sprint2\NoteBook5-Mva.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

def cell(cid, ctype, source):
    c = {'id': cid, 'cell_type': ctype, 'metadata': {}, 'source': source}
    if ctype == 'code':
        c['outputs'] = []
        c['execution_count'] = None
    return c

new_cells = [
    cell('a1b2c3d4', 'markdown', [
        '---\n',
        '# Partie 5 — Matrice de confusion et analyse critique des erreurs\n',
        '\n',
        '## 20. Prédictions et matrice de confusion'
    ]),
    cell('e5f6a7b8', 'code', [
        'from sklearn.metrics import confusion_matrix, classification_report\n',
        'import seaborn as sns\n',
        '\n',
        'y_pred_reg = model_reg.predict(X_test)\n',
        'y_pred_cls = np.argmax(y_pred_reg, axis=1)\n',
        'y_true_cls = y_test.flatten()\n',
        '\n',
        'cm = confusion_matrix(y_true_cls, y_pred_cls)\n',
        'per_class_acc = cm.diagonal() / cm.sum(axis=1)\n',
        '\n',
        'fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n',
        '\n',
        '# Matrice normalisée (taux)\n',
        'cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)\n',
        'sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",\n',
        '            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,\n',
        '            ax=axes[0], vmin=0, vmax=1)\n',
        'axes[0].set_title("Matrice de confusion normalisée (taux par classe réelle)")\n',
        'axes[0].set_xlabel("Prédit")\n',
        'axes[0].set_ylabel("Réel")\n',
        'plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")\n',
        '\n',
        '# Accuracy par classe triée\n',
        'sorted_idx = np.argsort(per_class_acc)\n',
        'colors_cls = ["tomato" if a < 0.75 else "steelblue" for a in per_class_acc[sorted_idx]]\n',
        'axes[1].barh([CLASS_NAMES[i] for i in sorted_idx], per_class_acc[sorted_idx], color=colors_cls)\n',
        'axes[1].axvline(per_class_acc.mean(), color="red", linestyle="--",\n',
        '                label=f"Moyenne {per_class_acc.mean():.3f}")\n',
        'axes[1].set_xlim(0, 1)\n',
        'axes[1].set_title("Accuracy par classe (triée, rouge < 75 %)")\n',
        'axes[1].set_xlabel("Accuracy")\n',
        'axes[1].legend()\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.show()\n',
        '\n',
        'print("\\nAccuracy par classe :")\n',
        'for i in sorted_idx:\n',
        '    print(f"  {CLASS_NAMES[i]:<12}: {per_class_acc[i]:.3f}")'
    ]),
    cell('c9d0e1f2', 'code', [
        '# Paires les plus confondues (hors diagonale)\n',
        'cm_off = cm.copy()\n',
        'np.fill_diagonal(cm_off, 0)\n',
        '\n',
        'pairs = []\n',
        'for i in range(10):\n',
        '    for j in range(10):\n',
        '        if i != j and cm_off[i, j] > 0:\n',
        '            pairs.append((cm_off[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))\n',
        'pairs.sort(reverse=True)\n',
        '\n',
        'cols = ["Rang", "Réel", "Prédit", "Erreurs", "% classe réelle"]\n',
        'print(f"{cols[0]:<5} {cols[1]:<14} {cols[2]:<14} {cols[3]:>8}  {cols[4]:>18}")\n',
        'print("-" * 65)\n',
        'for rank, (count, true_cls, pred_cls) in enumerate(pairs[:15], 1):\n',
        '    true_idx = CLASS_NAMES.index(true_cls)\n',
        '    pct = count / cm.sum(axis=1)[true_idx] * 100\n',
        '    print(f"{rank:<5} {true_cls:<14} {pred_cls:<14} {count:>8}  {pct:>17.1f}%")'
    ]),
    cell('d3e4f5a6', 'code', [
        '# Exemples visuels des 4 confusions les plus fréquentes\n',
        'top_pairs = [(pairs[i][1], pairs[i][2]) for i in range(4)]\n',
        '\n',
        'fig, axes = plt.subplots(4, 6, figsize=(14, 10))\n',
        '\n',
        'for row, (true_cls, pred_cls) in enumerate(top_pairs):\n',
        '    true_idx = CLASS_NAMES.index(true_cls)\n',
        '    pred_idx = CLASS_NAMES.index(pred_cls)\n',
        '    mask = (y_true_cls == true_idx) & (y_pred_cls == pred_idx)\n',
        '    err_indices = np.where(mask)[0][:6]\n',
        '    for col, idx in enumerate(err_indices):\n',
        '        ax = axes[row, col]\n',
        '        ax.imshow(X_test[idx])\n',
        '        ax.axis("off")\n',
        '        if col == 0:\n',
        '            ax.set_ylabel(f"Réel: {true_cls}\\n→ prédit: {pred_cls}",\n',
        '                          fontsize=8, rotation=0, labelpad=85, va="center")\n',
        '\n',
        'plt.suptitle("4 confusions les plus fréquentes — 6 exemples chacune", fontsize=12)\n',
        'plt.tight_layout()\n',
        'plt.show()'
    ]),
    cell('b7c8d9e0', 'code', [
        'print(classification_report(y_true_cls, y_pred_cls, target_names=CLASS_NAMES))'
    ]),
    cell('f1a2b3c4', 'markdown', [
        '## Analyse critique des erreurs\n',
        '\n',
        '### Confusions structurelles attendues\n',
        '\n',
        '| Paire confondue | Raison principale |\n',
        '|---|---|\n',
        '| **cat ↔ dog** | Formes corporelles proches (4 pattes, fourrure, oreilles) ; différences subtiles en 32×32 px |\n',
        '| **automobile ↔ truck** | Même sémantique visuelle (roues, carrosserie, vitre) ; seule la taille les distingue, illisible en basse résolution |\n',
        '| **deer ↔ horse** | Quadrupèdes à longues pattes, silhouettes similaires |\n',
        '| **bird ↔ airplane** | Même domaine visuel (ciel, ailes étendues, forme allongée) |\n',
        '\n',
        '### Pourquoi ces erreurs sont-elles inévitables à 32×32 ?\n',
        '\n',
        'CIFAR-10 est en basse résolution. Les détails discriminants sont perdus :\n',
        '- Les **textures fines** (poil vs peau lisse, rivets) disparaissent.\n',
        '- La **morphologie subtile** (museau du chien vs truffe du chat) se réduit à quelques pixels.\n',
        '- Le **contexte** (laisse, chenil) est souvent hors cadre ou illisible.\n',
        '\n',
        'Un humain aurait lui aussi un taux non nul sur ces images floues.\n',
        '\n',
        '### Ce que révèle la matrice\n',
        '\n',
        '**La matrice nest pas symétrique.** `cat → dog` peut être plus fréquent que `dog → cat` :\n',
        'le réseau a appris un biais en faveur de la classe aux features les plus saillantes.\n',
        '\n',
        '**Deux familles derreurs émergent :**\n',
        '- *Véhicules* : confusions intra-famille (automobile ↔ truck, ship ↔ airplane).\n',
        '- *Animaux* : sous-graphe dense — cat, dog, deer, horse se confondent mutuellement.\n',
        '\n',
        '### Pistes pour réduire ces erreurs\n',
        '\n',
        '| Levier | Effet attendu |\n',
        '|---|---|\n',
        '| Résolution plus élevée (64×64+) | Récupérer les détails texturaux discriminants |\n',
        '| Transfer learning (ResNet / EfficientNet ImageNet) | Réutiliser des features de haut niveau déjà apprises |\n',
        '| Label smoothing | Pénaliser moins les confusions proches, éviter la sur-confiance |\n',
        '| CutMix / MixUp | Forcer le modèle à ne pas sappuyer sur un seul patch image |'
    ])
]

nb['cells'].extend(new_cells)

with open(r'c:\indusense\Sprint2\NoteBook5-Mva.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Done -', len(nb['cells']), 'cells total')
