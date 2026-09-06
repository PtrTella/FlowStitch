# Capitolo 8: Analisi Sperimentale della Matrice e Posizionamento rispetto allo Stato dell'Arte (SOTA)

**Autore:** Pietro Tellarini  
**Framework:** FlowStitch (Zero-Shot Semantic Injection in Continuous Normalizing Flows)  
**Data di Validazione:** 5 Settembre 2026  
**Piattaforma HPC:** Cluster Giano (NVIDIA L40 GPU, 48GB VRAM)  

---

## 1. Risultati della Matrice Sperimentale: Evidenze Empiriche

L'esecuzione della matrice $4 \times 3 \times 2$ su FLUX.1-schnell ha prodotto le griglie comparative complete:
- **Sfera Blu**: [`outputs/experiment_cluster/cluster_matrix/a_blue_sphere/matrix_comparison_grid.png`](file:///Users/tella/Workspace/FlowStitch/outputs/experiment_cluster/cluster_matrix/a_blue_sphere/matrix_comparison_grid.png)
- **Cubo Rosso**: [`outputs/experiment_cluster/cluster_matrix/a_red_cube/matrix_comparison_grid.png`](file:///Users/tella/Workspace/FlowStitch/outputs/experiment_cluster/cluster_matrix/a_red_cube/matrix_comparison_grid.png)

```
                            COLONNE (Maschere di Supporto Spaziale)
               | pure_attention | calibrated_attention | fused_linear | hermite (C^1)
---------------+----------------+----------------------+--------------+---------------
R1: weld + kts | Solido         | Solido + Riflessi 🏆  | Trasparente  | Trasparente
R2: weld + cst | Solido         | Solido               | Trasparente  | Trasparente
R3: mos  + kts | Bordo Scuro    | Bordo Scuro          | Cratere      | Cratere
R4: mos  + cst | Bordo Scuro    | Bordo Scuro          | Cratere      | Cratere
R5: zero + kts | Noise Speckles | Noise Speckles       | Svanito      | Svanito
R6: zero + cst | Noise Speckles | Noise Speckles       | Svanito      | Svanito
```

---

## 2. Il Mistero della Densità di Hermite e del "Centro Svuotato"

### La Discrepanza tra Heatmap 2D e Integrazione ODE
In fase di test preliminare ([`fusion_suite_a_blue_sphere.png`](file:///Users/tella/Workspace/FlowStitch/outputs/exp_attention_cosine/fusion_suite_a_blue_sphere.png)), la formulazione Hermite $M = \text{smoothstep}(A \cdot \max(0, C))$ appariva visivamente come una sagoma circolare ben delineata. Tuttavia, nella generazione effettiva (Colonne 3 e 4), l'oggetto risulta **semitrasparente, acquoso o svuotato**.

L'indagine analitica e quantitativa ha rivelato due fenomeni fisici e percettivi congiunti:

#### A. L'Illusione Ottica della Colormap "Magma"
Nella visualizzazione matplotlib, la colormap percettiva assegna tonalità viola/magenta anche a valori molto bassi ($0.10 \le M \le 0.25$). L'occhio umano percepisce il contorno scuro su fondo nero e interpreta l'intera figura come "piena". 

#### B. Il Conflitto Vettoriale dei Riflessi Speculari nel Coseno
Sulla sfera blu 3D è presente una marcata riflessione speculare (la luce bianca in alto a sinistra):
- In corrispondenza della riflessione, i vettori di velocità $\mathbf{v}_0$ puntano in una direzione sostanzialmente ortogonale o opposta rispetto al corpo diffuso della sfera.
- La similarità coseno grezza $\cos(\theta) = \frac{\mathbf{v}_0 \cdot \bar{\mathbf{v}}}{\|\mathbf{v}_0\|\|\bar{\mathbf{v}}\|}$ crolla a valori fortemente negativi ($\approx -0.60$).
- Il clamp a zero $\max(0, \cos(\theta))$ e la successiva soglia di Hermite ($u = \text{clamp}\left(\frac{A \cdot C - 0.10}{0.65 - 0.10}\right)$) hanno azzerato **1452 pixel su 2958 all'interno del bounding box dell'oggetto**.
- Il valore medio della maschera nel nucleo dell'oggetto è crollato da **$0.62$ (Attention)** a **$0.09$ (Hermite)**!

#### C. L'Effetto nell'Integrazione ODE
Durante l'avanzamento differenziale:
$$\mathbf{v}_{\text{stitch}} = \mathbf{v}_{\text{lake}} + D(t) \cdot \lambda \cdot M(x) \cdot (\mathbf{v}_{\text{target}} - \mathbf{v}_{\text{lake}})$$
Se nel centro dell'oggetto $M(x) \approx 0.15$, allora il campo risultante è dominato per l'**85% dal vettore del lago** $\mathbf{v}_{\text{lake}}$ e solo per il **15% dal target**. Di conseguenza, la dinamica di FLUX ha sintetizzato acqua e fondale trasparente al posto del corpo opaco della sfera.

> **Conclusione**: `calibrated_attention` vince come maschera di supporto perché sottrae il piedistallo di rumore dello sfondo preservando la saturazione unitaria ($M \approx 1.0$) su tutto il volume solido interno.

---

## 3. Perché lo Sfondo Varia se il Seed è Identico?

Tutti i run sono stati generati rigorosamente con **`seed = 42`**, garantendo l'identità binaria del rumore ambientale $\mathbf{x}_0^{\text{lake}}$ al singolo bit. 

La variazione percepibile nel pattern dell'acqua e del fondale tra le colonne deriva dalla **Global Self-Attention bidirezionale (4096 $\times$ 4096)** del Transformer di FLUX:
1. In un'architettura DiT non esistono convoluzioni locali; ogni singolo token dell'acqua calcola il prodotto scalare Query-Key con tutti i token dell'immagine.
2. Quando al centro è presente una sfera blu opaca (Colonne 1 e 2), i token centrali modificano le matrici Key/Value globali: FLUX calcola l'occlusione della luce, la rifrazione caustica e le increspature superficiali che si propagano su tutto il lago.
3. Quando al centro la maschera si svuota (Colonne 3 e 4), l'attenzione globale "vede" un bacino d'acqua ininterrotto e calcola una distribuzione di luce continua sul fondale sabbioso.
4. La prova del determinismo è visibile confrontando Riga 1 (KTS) e Riga 2 (Constant) all'interno della stessa colonna: dove l'oggetto ha la stessa geometria, l'acqua circostante è **identica al pixel**.

---

## 4. Analisi Comparativa con lo Stato dell'Arte (SOTA)

### La Tassonomia dell'Editing e della Composizione

```
                                  PARADIGMI DI MANIPOLAZIONE GENERATIVA
                                                     |
        +--------------------------------------------+------------------------------------------+
        |                                                                                       |
[1. ATTENTION-INJECTION]                                                         [2. FLOW / ODE MANIPULATION]
(Stable Diffusion, DiT Early)                                                    (Continuous Normalizing Flows)
        |                                                                                       |
- Prompt-to-Prompt (Hertz '22)                                    +-----------------------------+-----------------------------+
- MasaCtrl (Cao '23)                                              |                                                           |
- Plug-and-Play (Tumanyan '23)                          [Inversion-Based]                                           [Inversion-Free]
- FREE-Edit ('24)                                                 |                                                           |
* Metodo: Hook forzato sui tensori K, V                 - RF-Inversion ('24)                                        - FlowEdit (Couairon '24)
  dentro i layer interni del modello.                     * Inversione numerica ODE all'indietro                      * Delta di 2 forward pass
* Limite: Viola l'ODE; dipendente da                      * Costo: 2x computazione,                                     (v_target - v_source) per
  un prompt target presente nel testo.                      instabilità numerica su pochi step.                         modifiche globali da testo.
                                                                                                                      |
                                                                                                            [FLOWSTITCH (Nostra Proposta)]
                                                                                                            * Zero-Shot Tangent Perturbation
                                                                                                            * Singolo forward pass in inferenza
                                                                                                            * Saldatura Sferica su S^D (Var=1)
                                                                                                            * KTS Damping per fisica ottica
                                                                                                            * Nessun tocco all'attenzione DiT
```

### Confronto Dettagliato

| Caratteristica | Prompt-to-Prompt / MasaCtrl | RF-Inversion (2024) | FlowEdit (2024) | **FlowStitch (Ours)** |
| :--- | :--- | :--- | :--- | :--- |
| **Architettura Target** | U-Net / Diffusione SDE | Rectified Flow (SD3/FLUX) | Rectified Flow (SD3/FLUX) | **Rectified Flow (FLUX.1)** |
| **Punto di Iniezione** | Matrici Key/Value Attention | Rumore latente invertito $x_1$ | Campo vettoriale differenziale | **Spazio Tangenziale ODE ($v_t$) + Rumore ($x_0$)** |
| **Modifica Pesi/Layer Interni**| **Sì** (Hook invasivi) | No | No | **No** (Rete completamente pura) |
| **Richiede Inversione ODE?** | Dipende (DDIM) | **Sì** (Risoluzione inversa) | No | **No** (Forward puro) |
| **Forward Pass per Step** | 1 | 1 (post inversione) | **2** ($v_{\text{tgt}} - v_{\text{src}}$) | **1** (Iniezione di $v_0$ offline) |
| **Composizione Multi-Scena** | Complessa / Conflitti | No (solo edit 1 immagine) | No (solo global re-prompt) | **Sì (Stitching zero-shot da DB vettoriale)** |
| **Conservazione Varianza** | Trascurata (clipping) | Euristica | Euristica | **Analitica ($\text{Var}=1$ su varietà $S^D$)** |
| **Armonizzazione di Confine** | Sfumatura gaussiana | Nessuna | Skip step uniformi | **KTS Damping $D(t) \to 0$** |

---

## 5. Il Valore Scientifico di FlowStitch per la Tesi

1. **Autonomia rispetto alla Cross-Attention**: Abbiamo dimostrato che non è necessario ingannare il modello forzando i token di testo nei blocchi interni. Il modello riconosce la presenza dell'oggetto attraverso la propria **Self-Attention visiva**, reagendo in modo fisicamente coerente (generando riflessi e onde).
2. **Efficienza Estrema**: Rispetto a *FlowEdit* (che richiede 2 valutazioni del modello DiT a 12B parametri ad ogni passo ODE) e a *RF-Inversion* (che richiede la traiettoria all'indietro), FlowStitch richiede **un solo forward pass per step**, attingendo ai vettori di velocità precalcolati offline nel database.
3. **Fondamento Differenziale**: L'iniezione nello spazio tangente con saldatura sferica e smorzamento KTS costituisce un framework matematicamente chiuso e rigoroso per il controllo geometrico nei Continuous Normalizing Flows.
