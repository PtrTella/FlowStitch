# FlowStitch: Differential Trajectory Routing in Continuous Normalizing Flows
**Zero-Shot Semantic Injection via Tangent-Space Perturbation**

## Abstract
Questo documento formalizza il framework teorico di **FlowStitch**, una metodologia per l'iniezione semantica zero-shot in modelli generativi basati su *Continuous Normalizing Flows* (es. FLUX.1). Il documento decostruisce matematicamente i limiti dei tradizionali approcci di *spatial inpainting* (Masking Statistico), dimostrando l'incompatibilità intrinseca tra singolarità topologiche e i solutori ODE in regimi *Few-Step*. Come soluzione, formalizziamo la **Perturbazione nello Spazio Tangente (ODE Perturbation)**: un'iniezione fluida guidata dalla Cross-Attention che garantisce coerenza termodinamica universale con complessità $\mathcal{O}(N)$.

---

## 1. Introduzione e Contesto Teorico
I recenti sviluppi nei *Continuous Normalizing Flows* e *Rectified Flows* (Liu et al., 2022; Lipman et al., 2023) hanno ridefinito la generazione di immagini mappando distribuzioni di rumore a distribuzioni di dati tramite Equazioni Differenziali Ordinarie (ODE) con traiettorie quasi-lineari:
$$ x_t = t x_1 + (1-t) x_0 $$
$$ v_t = \frac{dx_t}{dt} = x_1 - x_0 $$

L'adattamento delle tecniche di Inpainting classico (che storicamente operano tramite mascherature binarie in spazio pixel o latente) a questo framework differenziale presenta severe sfide. Tentativi precedenti come *Prompt-to-Prompt* (Hertz et al., 2022) hanno dimostrato la potenza della manipolazione della *Cross-Attention*, ma l'iniezione semantica di soggetti volumetrici isolati richiede un controllo superiore sul campo vettoriale generativo.

---

## 2. Fase 1: Limiti dell'Inpainting Spaziale (Statistical Masking)
L'approccio esplorativo di FlowStitch mirava a isolare i tensori vettoriali di un soggetto (Target) applicando tecniche di segmentazione sull'energia cinetica del modello.

### 2.1 Formulazione del Gating Termodinamico
Definita l'Energia Cinetica del gradiente latente iniziale come la norma $L_2$:
$$ E(x,y) = ||v_{0}(x,y)||_2 $$

Abbiamo postulato l'estrazione di una mappa continua ($\alpha_{gated} \in [0,1]$) applicando una normalizzazione e una funzione ReLU traslata. La soglia di taglio $\tau$ è stata derivata statisticamente (Diseguaglianza di Čebyšëv):
$$ \tau = \mu_E + \sigma_E $$
$$ \alpha_{gated} = \frac{\max(0, \alpha_{raw} - \tau)}{1 - \tau} $$

L'obiettivo era ottenere una fusione spaziale esatta: $v_{stitch} = v_{ambient} \odot (1 - \alpha_{gated}) + v_{target} \odot \alpha_{gated}$.

### 2.2 Dinamiche di Fallimento nei Sistemi ODE
L'Ablation Study su geometrie primitive ha rivelato due patologie sistemiche che invalidano l'approccio spaziale:

1. **Il Vuoto Termodinamico Pervasivo**: Superfici geometricamente omogenee (centro di una sfera, facce piane di un cubo) esigono una variazione vettoriale marginale. Di conseguenza, l'energia $E(x,y)$ locale collassa strutturalmente sotto la soglia statistica $\tau$. Il gating confonde la stabilità vettoriale per "rumore di fondo", generando maschere svuotate internamente (maschere anulari).
2. **Discontinuità e Incompatibilità Few-Step**: Sui contorni, il gating genera gradienti frammentati. L'imposizione di una maschera approssimabile a una funzione gradino ($\mathcal{H}$) introduce discontinuità violente nel campo di velocità. Un integratore di Eulero a bassissima risoluzione (es. 4 step) non possiede la granularità temporale per interpolare salti vettoriali così drammatici, causando collassi geometrici e aloni fantasma (*Ghosting*).

*(Relazione Teorica: Lo Spectral Matting basato sul Laplaciano dei Grafi [Levin et al., 2008] tramite l'estrazione del Vettore di Fiedler mitiga parzialmente i Vuoti Termodinamici, tuttavia la sua complessità $\mathcal{O}(N^3)$ ne preclude l'utilizzo in architetture High Performance Computing).*

---

## 3. Fase 2: Il Paradigma della Perturbazione ODE Spazio-Tangente
Per preservare la fluidità $C^1$ intrinseca al Flow Matching, FlowStitch abbandona il concetto di *masking spaziale* in favore di una **Perturbazione Differenziale nel Dominio del Tempo**.

### 3.1 Equazione della Perturbazione
Invece di sovrascrivere porzioni di tensore, induciamo un campo di forza che devia morbidamente la derivata temporale dell'ambiente verso l'attrattore topologico del target.
Utilizziamo la *Cross-Attention Matrix* nativa come mappa di densità probabilistica spaziale. Dato il token target, si estrae la mappa media $A_{target} \in [0,1]$ e si applica:

$$ v_{stitch} = v_{ambient} + \lambda \left[ A_{target} \odot (v_{target} - v_{ambient}) \right] $$

Dove:
- $(v_{target} - v_{ambient})$ è il differenziale vettoriale puro ($\Delta v$).
- $\odot$ rappresenta il Prodotto di Hadamard (Element-wise).
- $\lambda$ è il fattore di iper-compensazione spaziale.

### 3.2 Invarianza Topologica (Zero-Shot Fluid Dynamics)
L'approccio *Threshold-Free* risolve nativamente le patologie della Fase 1. L'analisi della magnitudo della forza iniettata ($||\Delta v||_2$) dimostra un adattamento strutturale perfetto (Ablation su primitive multiple):
- **Radiale**: Su forme sferiche l'energia colma nativamente i centri (estirpazione del Vuoto Termodinamico).
- **Ortogonale**: Su spigoli netti (Cubi), la rete intensifica la forza in modo spontaneo per rompere la fluidità ambientale senza tagliare.
- **Diagonale/Acuto**: Sulle piramidi, la forza traccia le generatrici.
L'algoritmo opera a complessità rigorosamente $\mathcal{O}(N)$.

---

## 4. Roadmap di Integrazione HPC
La formulazione matematica sopra definita guiderà lo sviluppo del kernel di produzione (`flux_latent_stitching.py`) sull'infrastruttura High Performance. I passi architetturali includono:

1. **Routing Topologico 2D**: Sviluppo di operatori di *Affine Transform* o *Tensor Rolling* per traslare l'influenza semantica di $A_{target}$ e $v_{target}$ verso coordinate arbitrarie del batch ambientale.
2. **Calibrazione Autoregressiva di $\lambda$**: Derivazione dinamica dell'iper-parametro di spinta gravitazionale in funzione dell'inverso dell'integrale di superficie dell'Attenzione, per scalare l'efficacia su soggetti di dimensione marginale.
3. **End-to-End Latent Execution**: Validazione della stabilità del Flow Matching perturbato tramite VAE decoding a pipeline completa sui nodi GPU.

---

### Referenze
* Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow matching for generative modeling. *ICLR*.
* Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR*.
* Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., & Cohen-Or, D. (2022). Prompt-to-prompt image editing with cross attention control. *ICLR*.
* Levin, A., Rav-Acha, A., & Lischinski, D. (2008). Spectral matting. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
