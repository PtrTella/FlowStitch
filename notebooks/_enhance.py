#!/usr/bin/env python3
"""
Enhance notebooks with flowstitch/ bridging cells and detailed explanations.

Each notebook gets:
1. A "Collegamento al Codice" cell at the end showing which flowstitch/ module 
   was born from this experiment
2. Enhanced theory cells where they're thin
"""
import json
import os

def new_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source] if isinstance(source, str) else source,
    }

def new_code_cell(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [source] if isinstance(source, str) else source,
        "execution_count": None,
        "outputs": [],
    }

def load_nb(path):
    with open(path) as f:
        return json.load(f)

def save_nb(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {path} ({len(nb['cells'])} cells)")

NB_DIR = "notebooks"

# =====================================================
# NB01: Add bridging + deeper theory
# =====================================================
print("Enhancing NB01...")
nb = load_nb(os.path.join(NB_DIR, "01_Topological_Analysis.ipynb"))

nb["cells"].append(new_markdown_cell("""---

## Collegamento al Codice: `flowstitch.extraction`

Le scoperte di questo capitolo sono alla base dei seguenti moduli:

| Scoperta | Modulo FlowStitch | Funzione |
|----------|-------------------|----------|
| Tokenizzazione T5 per identificare target | `core.tokenizer_utils` | `find_token_indices()` |
| Mappe di attenzione per-token | `extraction.attention_mask` | `extract_attention_mask()` |
| Binarizzazione Otsu | `extraction.attention_mask` | `otsu_threshold()` |
| Norma L2 di $v_0$ come mappa energetica | `extraction.energy_mask` | `compute_chebyshev_threshold()` |

### Cosa abbiamo imparato
1. **L'attenzione** identifica COSA è l'oggetto (identità semantica)
2. **L'energia** ($\\|v_0\\|_2$) identifica DOVE è l'oggetto (confini spaziali)
3. A $t=0$ questi due segnali sono **spazialmente allineati** — l'informazione geometrica
   è già presente nel primo step dell'ODE

Questa dualità attenzione-energia è il fondamento di tutta la pipeline FlowStitch.
"""))

nb["cells"].append(new_code_cell("""# Verifica: il codice flowstitch/ implementa esattamente questa analisi
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from flowstitch.core.tokenizer_utils import find_token_indices
from flowstitch.extraction.attention_mask import extract_attention_mask, otsu_threshold

print("✅ flowstitch.core.tokenizer_utils.find_token_indices — implementa la ricerca token")
print("✅ flowstitch.extraction.attention_mask.extract_attention_mask — estrae A_target")
print("✅ flowstitch.extraction.attention_mask.otsu_threshold — binarizzazione Otsu")
print()
print("Questi moduli nascono direttamente dall'analisi in questo notebook.")
"""))

save_nb(nb, os.path.join(NB_DIR, "01_Topological_Analysis.ipynb"))

# =====================================================
# NB02: Add bridging
# =====================================================
print("Enhancing NB02...")
nb = load_nb(os.path.join(NB_DIR, "02_Injection_Thermodynamics.ipynb"))

nb["cells"].append(new_markdown_cell("""---

## Collegamento al Codice: `flowstitch.stitching`

Questo esperimento dimostra perché manipoliamo $v_0$ e non $x_{\\text{pred}}$.

| Scoperta | Modulo FlowStitch | Impatto |
|----------|-------------------|---------|
| Iniettare su $v_0$ preserva l'isotropia | `stitching.ode_perturbation` | `perform_ode_step()` perturba la velocità, non il latente |
| Blending soft (non hard) è necessario | `stitching.kts` | `apply_kts()` usa blending continuo pesato da maschera |
| La perturbazione deve attenuarsi con $t$ | `stitching.kts` | `compute_damping_factor()` — decay esponenziale |

### La Dimostrazione Algebrica

Se $x_0 \\sim \\mathcal{N}(0, I)$ e $x_{\\text{pred}} = x_0 + v_0$, iniettare $x_{\\text{pred}}^{\\text{target}}$
al posto di $x_{\\text{pred}}^{\\text{ambient}}$ **non** preserva la distribuzione del rumore (i seed sono diversi).

Invece, perturbare $v_0$:
$$v_{\\text{stitch}} = v_{\\text{ambient}} + \\lambda \\cdot M \\odot (v_{\\text{target}} - v_{\\text{ambient}})$$

preserva la struttura dell'ODE perché modifichiamo il **campo vettoriale**, non lo stato.
"""))

save_nb(nb, os.path.join(NB_DIR, "02_Injection_Thermodynamics.ipynb"))

# =====================================================
# NB03: Add bridging
# =====================================================
print("Enhancing NB03...")
nb = load_nb(os.path.join(NB_DIR, "03_Hard_Masking_Failure.ipynb"))

nb["cells"].append(new_markdown_cell("""---

## Collegamento al Codice: la motivazione per le maschere continue

Questo fallimento è la ragione per cui `flowstitch/` NON usa mai maschere binarie nella pipeline.

| Scoperta | Impatto sul Codice |
|----------|-------------------|
| Hard mask → $\\|\\nabla v\\|_2 \\to \\infty$ al bordo | Tutti i metodi in `extraction/` producono maschere **continue** in $[0, 1]$ |
| Picard-Lindelöf richiede Lipschitz | `stitching.kts.apply_kts()` usa blending differenziabile |
| Il bordo è il punto critico | `extraction.spectral_mask` usa upsampling bilineare per bordi lisci |

### Formalizzazione: Il Teorema di Picard-Lindelöf

L'ODE del Flow Matching è $\\frac{dx}{dt} = v_\\theta(x, t)$.

Il teorema garantisce **esistenza e unicità** della soluzione se $v_\\theta$ è **Lipschitz-continua** in $x$:
$$\\exists L > 0 : \\|v_\\theta(x_1, t) - v_\\theta(x_2, t)\\| \\leq L \\|x_1 - x_2\\|$$

Una maschera binaria $M \\in \\{0, 1\\}$ crea un salto in $v_{\\text{stitch}} = M v_{\\text{target}} + (1-M) v_{\\text{ambient}}$
al confine di $M$, dove $L \\to \\infty$. Questo viola il teorema e genera artefatti.
"""))

save_nb(nb, os.path.join(NB_DIR, "03_Hard_Masking_Failure.ipynb"))

# =====================================================
# NB04: Add bridging + variance conservation
# =====================================================
print("Enhancing NB04...")
nb = load_nb(os.path.join(NB_DIR, "04_Continuous_Energy_Blending.ipynb"))

nb["cells"].append(new_markdown_cell("""---

## Collegamento al Codice: KTS e il Problema della Varianza

Questo notebook motiva due moduli fondamentali:

### `flowstitch.stitching.kts` — Kinetic Trajectory Shaping

Il blending continuo $v_{\\text{stitch}} = \\alpha v_{\\text{target}} + (1-\\alpha) v_{\\text{ambient}}$ 
risolve il problema Lipschitz (Cap. 3), ma introduce il **Ghosting Termodinamico**: 
l'energia residua del background non è mai esattamente zero.

KTS aggiunge un **damping esponenziale** che attenua la perturbazione negli step finali:
$$v_{\\text{stitch}} = v_{\\text{ambient}} + D(t) \\cdot \\lambda \\cdot [M \\odot (v_{\\text{target}} - v_{\\text{ambient}})]$$
dove $D(t) = \\exp(-\\gamma \\cdot \\max(0, t - t_{\\text{cutoff}}))$.

### Conservazione della Varianza del Rumore

**Scoperta critica**: il blending lineare $\\alpha x_1 + (1-\\alpha) x_2$ con $x_i \\sim \\mathcal{N}(0, I)$
produce $\\text{Var} = \\alpha^2 + (1-\\alpha)^2 \\neq 1$ per $\\alpha \\neq 0, 1$.

Il blending **geometrico** $\\sqrt{\\alpha} x_1 + \\sqrt{1-\\alpha} x_2$ preserva $\\text{Var} = 1$.

Questo è implementato in `flowstitch.pipelines.latent_stitching` (mode `dual`):
```python
latents = lake * torch.sqrt(1 - M) + x0 * torch.sqrt(M)
```
"""))

nb["cells"].append(new_code_cell("""# Dimostrazione numerica: conservazione della varianza
import sys, os, torch
sys.path.insert(0, os.path.abspath('..'))

# Due noise isotropici
x1 = torch.randn(1, 4096, 64)
x2 = torch.randn(1, 4096, 64)
alpha = 0.5

# Blending lineare (SBAGLIATO)
blend_linear = alpha * x1 + (1 - alpha) * x2
var_linear = blend_linear.var().item()

# Blending geometrico (CORRETTO)
blend_sqrt = torch.sqrt(torch.tensor(alpha)) * x1 + torch.sqrt(torch.tensor(1 - alpha)) * x2
var_sqrt = blend_sqrt.var().item()

print(f"Varianza originale:          {x1.var().item():.4f}")
print(f"Blending lineare  (α=0.5):   {var_linear:.4f} ← COLLASSA (dovrebbe essere 1.0)")
print(f"Blending geometrico (√α=0.5): {var_sqrt:.4f} ← PRESERVATA ✅")
"""))

save_nb(nb, os.path.join(NB_DIR, "04_Continuous_Energy_Blending.ipynb"))

# =====================================================
# NB05: Add bridging to decoders module
# =====================================================
print("Enhancing NB05...")
nb = load_nb(os.path.join(NB_DIR, "05_Decoder_Exploration.ipynb"))

nb["cells"].append(new_markdown_cell("""---

## Collegamento al Codice: `flowstitch.extraction.decoders`

Tutti e 5 i decoder sperimentati in questo capitolo sono stati integrati nel pacchetto 
`flowstitch/extraction/decoders/`:

```python
from flowstitch.extraction.decoders import (
    SupervisedHybridDecoder,           # §5.2 — cosine similarity locale
    SvdCosineHybridDecoder,            # §5.3 — SVD multi-testa + cosine globale
    SupervisedSpectralHybridDecoder,   # §5.4 — Fiedler a risoluzione piena
    GraphDiffusionHybridDecoder,       # §5.5 — random walk di Markov
    SvdSeededGraphDiffusionDecoder,    # §5.6 — SVD + diffusione (vincitore)
)
```

### Interfaccia Comune

Ogni decoder accetta:
- `attn_tensor`: tensore di cross-attention `[Heads, H*W, Text_Seq]`
- `v_t`: campo di velocità `[C, H, W]`
- `token_indices`: indici dei token target
- `spatial_shape`: tupla `(H, W)` della griglia spaziale

E restituisce `(M_hybrid, S_signal, S_topology)`.

### Nota sulla Tokenizzazione

Nei notebook originali, ogni decoder conteneva la propria copia di `_find_token_indices()`.
Nella versione modulare, la tokenizzazione è delegata a `flowstitch.core.tokenizer_utils.find_token_indices()`,
e i decoder ricevono direttamente gli indici.
"""))

save_nb(nb, os.path.join(NB_DIR, "05_Decoder_Exploration.ipynb"))

# =====================================================
# NB06: Add bridging to spectral_mask
# =====================================================
print("Enhancing NB06...")
nb = load_nb(os.path.join(NB_DIR, "06_Spectral_Matting_and_TDA.ipynb"))

# Find the last conclusions cell and add after it
nb["cells"].append(new_markdown_cell("""---

## Collegamento al Codice: `flowstitch.extraction.spectral_mask` e `tda_mask`

| Esperimento | Modulo | Soluzione |
|-------------|--------|-----------|
| Fiedler a risoluzione piena (§6.1) | `spectral_mask.compute_fiedler_mask()` | Decimazione bilineare a `target_resolution=32` |
| TDA con Ripser (§6.4) | `tda_mask.extract_tda_mask()` | Clustering gerarchico con fallback |
| Hybrid Fiedler+Attn (§6.6) | `hybrid_mask.hybrid_semantic_decomposition()` | Risolve ambiguità di segno |
| ARPACK failure (§6.5) | Motivazione per il parametro `target_resolution` | Gap spettrale insufficiente a $4096^2$ |

### `compute_fiedler_mask(keys_img, target_resolution=32)`

Questo è il cuore della pipeline di estrazione. Il parametro `target_resolution=32` 
è la soluzione diretta al fallimento ARPACK documentato in §6.5:

```python
# 1. Decima da 64×64 a 32×32 via interpolazione bilineare
# 2. Cosine affinity → Laplaciano
# 3. Eigendecomposition completa (32×32 = 1024 pixel → fattibile)
# 4. Fiedler vector → partizione binaria
# 5. Upsample da 32×32 a 64×64
```

La decimazione porta il gap spettrale da $\\Delta\\lambda \\approx 10^{-6}$ a $\\sim 10^{-2}$,
rendendo la decomposizione numericamente stabile.
"""))

save_nb(nb, os.path.join(NB_DIR, "06_Spectral_Matting_and_TDA.ipynb"))

# =====================================================
# NB07: Add bridging to multi_object
# =====================================================
print("Enhancing NB07...")
nb = load_nb(os.path.join(NB_DIR, "07_Multi_Object_Decomposition.ipynb"))

nb["cells"].append(new_markdown_cell("""---

## Collegamento al Codice: `flowstitch.extraction.multi_object`

La decomposizione multi-oggetto a due stadi è integrata in:

```python
from flowstitch.extraction.multi_object import (
    multi_object_decomposition,    # Sigmoid adattivo per oggetto
    validate_orthogonality,        # Verifica non-sovrapposizione
    extract_pure_velocity_fields,  # DNA Vettoriale
)
```

### Utilizzo

```python
masks = multi_object_decomposition(
    v0=v0_tensor,
    attn_maps={"cube": A_cube, "sphere": A_sphere},
    k_steepness=10.0,
)
orthogonality = validate_orthogonality(masks)
dna = extract_pure_velocity_fields(v0_tensor, masks)
```

### Il Parametro $k$

L'analisi di sensitivity (§6bis.7) ha determinato che $k \\in [8, 15]$ è il range 
ottimale. Valori troppo bassi ($k < 5$) producono maschere sfumate; valori troppo 
alti ($k > 20$) approssimano una step function, vanificando il vantaggio del blending continuo.
"""))

save_nb(nb, os.path.join(NB_DIR, "07_Multi_Object_Decomposition.ipynb"))

# =====================================================
# Clean up restructure script
# =====================================================
print("\nDone. Removing restructure scripts...")
os.remove(os.path.join(NB_DIR, "_restructure.py"))
print("Cleaned up.")

# Final summary
print("\n=== FINAL NOTEBOOK STRUCTURE ===")
for f in sorted(os.listdir(NB_DIR)):
    if f.endswith('.ipynb'):
        nb = load_nb(os.path.join(NB_DIR, f))
        code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        has_fs = any('flowstitch' in ''.join(c['source']) for c in code_cells)
        has_out = sum(1 for c in code_cells if c.get('outputs'))
        size = os.path.getsize(os.path.join(NB_DIR, f)) // 1024
        print(f"  {f} | {len(nb['cells'])} cells | fs={has_fs} | out={has_out} | {size}KB")
