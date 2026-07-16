# FlowStitch: Un Framework per la Decomposizione Latente Non Supervisionata tramite Campi di Velocità e Analisi Spettrale dell'Attenzione

**Autore:** Pietro Tellarini  
**Data:** 13 Maggio 2026  
**Area di Ricerca:** Modelli Generativi, Flow Matching, Computer Vision, Trasporto Ottimale

---

## Abstract

La manipolazione generativa di scene multi-oggetto richiede una decomposizione latente ad altissima fedeltà, capace di isolare entità sovrapposte e semanticamente contigue. Questo studio presenta **FlowStitch**, un framework teorico e computazionale che estende il paradigma del Flow Matching (FM) oltre la generazione, elevandolo a strumento di indagine topologica e segmentazione non supervisionata. Mentre i metodi zero-shot allo stato dell'arte si basano sull'analisi statica delle mappe di cross-attention o su filtri energetici scalari empirici—afflitti da instabilità, "token overflow" e cecità geometrica—FlowStitch introduce la **Topological Data Analysis (TDA)** nello spazio latente. Calcolando l'**Omologia Persistente** del campo di velocità, identifichiamo gli oggetti non come aggregati di energia, ma come **invarianti topologici (Betti $H_0, H_1$)**. Questa riformulazione risolve in modo esatto e parameter-free il paradosso del *"Vuoto Termodinamico"*, consentendo il **Kinetic Trajectory Shaping (KTS)** per l'iniezione multi-oggetto stabile e Lipschitz-continua.

---

## 1. Introduzione

L'avvento dei modelli generativi basati su Flow Matching (Lipman et al., 2023) ha introdotto un rigore analitico superiore rispetto ai precedenti modelli Diffusion-based, modellando il processo generativo come il trasporto continuo di una distribuzione di probabilità. Tuttavia, la capacità di effettuare **decomposizione latente non supervisionata**—ovvero separare un tensore complesso nei suoi "atomi" semantici e fisici senza l'ausilio di reti esterne—rimane un problema irrisolto.

I metodi contemporanei, come *Diffuse Attend and Segment* (Tian et al., 2024) e *Seg4diff* (Kim et al., 2026), si fondano sull'ipotesi che le matrici di cross-attention contengano la planimetria completa della scena. Purtroppo, l'attenzione è un costrutto vettoriale "statico": satura rapidamente in presenza di oggetti occlusi o semanticamente simili (ad es. "un cilindro rosso dietro una sfera rossa").

In questo lavoro proponiamo una rottura epistemologica: **la struttura spaziale non risiede in ciò che il modello "guarda" (attenzione), ma in come il modello "spinge" la massa probabilistica (velocità)**. 

I contributi di questa ricerca sono:
1.  **Dimostrazione analitica del Vuoto Termodinamico**: Formalizzazione del limite intrinseco dei descrittori puramente basati sulla magnitudo dell'energia cinetica.
2.  **Il Decoder Ibrido FlowStitch**: Una sintesi algoritmica tra l'analisi spettrale delle mappe di attenzione e la similarità coseno del campo vettoriale.
3.  **Kinetic Trajectory Shaping (KTS)**: Una tecnica di manipolazione delle traiettorie ODE che aggira le singolarità terminali, preservando la costante di Lipschitz del campo ibrido.

---

## 2. Fondamenti Teorici e Analisi dello Stato dell'Arte

### 2.1 Flow Matching come Problema Inverso di Fluidodinamica
Il Continuous Normalizing Flow (CNF) alla base del Flow Matching modella l'evoluzione di una densità $p_t$ tramite l'equazione di continuità associata a un campo di velocità tempo-dipendente $v_\theta(x,t)$. La connessione con la teoria del Trasporto Ottimale è radicata nella formulazione dinamica di **Benamou-Brenier** (2000):
$$ W_2^2(p_0, p_1) = \inf_{(v, p)} \int_0^1 \int_{\mathbb{R}^d} \|v_t(x)\|^2 p_t(x) dx dt $$
A differenza di Li et al. (2026), che utilizzano l'energia per definire metriche di fedeltà macroscopica, FlowStitch opera una *micro-analisi locale* su $\Omega \subset \mathbb{R}^d$, analizzando la deformazione istantanea per isolare componenti dell'immagine.

### 2.2 Kinetic Path Energy (KPE) e il Dualismo Energia-Densità
Recentemente, la letteratura ha esplorato l'uso dell'energia per ispezionare il manifold generativo (*EnfoPath*, Li et al., 2025). Definiamo la **Kinetic Path Energy** per una traiettoria spaziale puntuale $z(t)$:
$$ E(z) = \frac{1}{2} \int_0^1 \|v_\theta(z(t), t)\|^2 dt $$
Il Teorema della Relazione Energia-Densità dimostra che:
$$ \|v_\theta(z, t)\|^2 \asymp -\nabla_z \log \hat{p}_t(z) $$
Ciò implica una legge fondamentale: **l'energia cinetica raggiunge i suoi massimi locali dove la densità di probabilità subisce la massima variazione (i bordi fisici degli oggetti)**, poiché il modello deve esercitare un'enorme "spinta" per distaccarsi dal rumore di fondo e posizionare una struttura coerente. Al contrario, all'interno di regioni piatte e omogenee (dove la densità del target locale è alta e stabile), l'energia crolla verso lo zero.

### 2.3 Scomposizione di Helmholtz-Hodge
Il campo $v_\theta$ appreso non è irrotazionale. Può essere scomposto secondo il teorema di Helmholtz-Hodge:
$$ v_t = \nabla \Phi + \nabla \times \mathbf{A} $$
Mentre il potenziale scalare $\Phi$ governa il collasso del rumore gaussiano verso il dominio dell'immagine naturale (flusso base), il potenziale vettore $\mathbf{A}$ codifica la **vorticità semantica**, ovvero le perturbazioni locali necessarie per formare oggetti discreti contro il background. È proprio studiando la discontinuità di queste derivate direzionali che FlowStitch supera la cecità delle mappe di attenzione.

---

## 3. Dal Paradosso Energetico all'Omologia Persistente

### 3.1 Il "Vuoto Termodinamico" e la Crisi delle Soglie Scalari
Il tentativo di utilizzare soglie statistiche sull'energia (ad es. $E(z) > \mu + \sigma$) per estrarre maschere di segmentazione porta a un artefatto sistematico. Poiché nel centro (bulk) dell'oggetto l'energia vettoriale tende a zero (in virtù della stabilità termodinamica), un filtro energetico scava l'oggetto al suo interno, producendo **maschere anulari**. Chiamiamo questa patologia *Vuoto Termodinamico*.
I tentativi di risolvere questo vuoto incrociando l'energia con la Cross-Attention tramite gating non-lineari (Sigmoidi con parametri $\lambda, k$) hanno mostrato un'estrema instabilità ai parametri, rendendo impossibile una generalizzazione zero-shot robusta.

### 3.2 Topological Data Analysis (TDA) del Flusso Latente
Per trascendere i limiti locali (pixel-wise) e scalari, FlowStitch proietta il campo vettoriale in un framework di **Topologia Algebrica**. Se modelliamo il campo $v_t$ come una nube di punti pesata dalla Similarità Coseno direzionale, gli oggetti fisici smettono di essere distribuzioni di magnitudo e diventano **invarianti topologici (complessi simpliciali)**.

**1. Filtrazione di Vietoris-Rips e Numeri di Betti:**
Calcolando l'Omologia Persistente sul campo di velocità deformato, ricaviamo i Numeri di Betti:
- **$H_0$ (Componenti Connesse)**: Determina esplicitamente il numero di entità isolate (gli oggetti target), senza necessità di soglie energetiche.
- **$H_1$ (Cicli)**: Permette la risoluzione rigorosa delle occlusioni spaziali.

**2. Il Diagramma di Persistenza:**
Un oggetto fisico reale (target) possiede una firma inequivocabile nel diagramma di persistenza: nasce presto (bassa distanza geodetica) e muore molto tardi (alta coesione vettoriale). Il rumore termodinamico di background si annienta lungo la diagonale (Birth $\approx$ Death). L'estrazione della maschera latente $\alpha$ si riduce al banale isolamento dei generatori $H_0$ con massima persistenza.

Questa svolta topologica fornisce una soluzione **esatta e parameter-free** all'isolamento multi-oggetto.

### 3.3 Kinetic Trajectory Shaping (KTS)
A differenza dei metodi di *image stitching* standard, FlowStitch opera un intervento chirurgico durante l'integrazione ODE. Perturbiamo la derivata tangenziale al tempo $t$:
$$ \frac{dz}{dt}_{new} = v_{ambient}(t) + M_{hybrid} \odot (v_{target}(t) - v_{ambient}(t)) \cdot D(t) $$
L'uso del fattore di damping $D(t) = e^{-\gamma (t - t_{cutoff})_+}$ implementa un **Kinetic Soft-Landing**. Questo è fondamentale: la letteratura (Li et al., 2026) dimostra che l'eccesso di energia cinetica per $t \to 1$ causa "Collisione di Memorizzazione" (il modello replica fedelmente gli artefatti del training set). Smorzando la traiettoria, FlowStitch forza il modello ad armonizzare l'oggetto innestato con le statistiche locali del nuovo background. Inoltre, poiché $M_{hybrid}$ è $C^1$-continua, l'equazione differenziale rimane stabile per solver a basso step (e.g., Euler).

---

## 4. Discussione e Implicazioni Teoriche

La metodologia FlowStitch mette radicalmente in discussione l'assunto per cui le matrici di attenzione siano l'unica metrica di "intenzionalità" in una rete generativa multimodale. Mentre approcci come *Diffuse Attend and Segment* si sforzano di pulire l'attenzione con complesse reti ausiliarie, FlowStitch dimostra che l'invarianza spaziale risiede nativamente nell'**Omologia del campo vettoriale**, mentre l'attenzione serve solo come prior concettuale.

FlowStitch risolve matematicamente l'ambiguità dell'occlusione tramite la Topologia Algebrica: due sfere identiche sovrapposte genereranno la medesima attenzione, ma costituiranno generatori omologici separati (due $H_0$ distinti separati da una barriera di divergenza), facilmente isolati tramite Diagrammi di Persistenza.

---

## 5. Conclusioni e Sviluppi Futuri

Questa ricerca stabilisce un nuovo paradigma diviso in fasi esplorative:
1.  **Dinamica Generativa (Notebooks 1-2):** Definizione del flow matching tramite l'equazione di continuità e il Trasporto Ottimale.
2.  **Crisi dei Descrittori Scalari (Notebooks 3-5):** Analisi del fallimento delle hard mask e scoperta del "Vuoto Termodinamico".
3.  **Il Limite dell'Euristica Globale (Notebook 6):** L'utilizzo di un gating termodinamico parametrico ha dimostrato di non poter risolvere le patologie spaziali (frammentazione topologica e bleeding) a causa della natura puramente scalare dell'energia.
4.  **Scomposizione Semantico-Vettoriale e Graph Cuts (Notebook 7):** Risoluzione definitiva tramite una complessa pipeline matematica:
    - *Sub-Token Resolution:* Soluzione architetturale al problema dell'allineamento testo-immagine nei modelli BPE, calcolando l'attenzione aggregata per ricomporre la semantica frammentata.
    - *Spectral Co-Embedding Graph Cut:* Abbandono del banale thresholding in favore di un grafo spaziale i cui pesi bilanciano l'affinità di Cross-Attention e la Similarità Coseno del flusso. Il Normalized Cut (tramite il vettore di Fiedler del Laplaciano) permette di spezzare il grafo esattamente lungo i confini fisici di oggetti semanticamente complessi.
    - *Estrazione del Generative Prior:* L'output della scomposizione non è una semplice maschera d'immagine ($\alpha$), ma l'estrapolazione di $\tilde{v}_0 = \alpha \odot v_0$. Questo tensore rappresenta il *comando di denoising vettoriale* dell'oggetto, pronto per essere archiviato in un database e iniettato per condizionare nativamente future rigenerazioni (Flow Stitching).

Questa fondazione solida apre la strada a future esplorazioni avanzate come il **4D Video Flow Matching** (i target diventano tubi topologici spazio-temporali $H_1$) e l'introduzione di **Attrattori Hamiltoniani Cross-Modali** per deformare le traiettorie generative sfruttando il Teorema di Liouville.

---

## Riferimenti Bibliografici

1.  **Lipman, Y., et al. (2023).** *Flow Matching for Generative Modeling.* ICLR 2023. arXiv:2210.02747.
2.  **Li, Z., et al. (2026).** *A Kinetic-Energy Perspective of Flow Matching.* arXiv:2602.07928.
3.  **Li, Z., et al. (2025).** *EnfoPath: Energy-Informed Analysis of Generative Trajectories.* arXiv:2511.19087.
4.  **Carlsson, G. (2009).** *Topology and Data.* Bulletin of the American Mathematical Society, 46(2), 255-308.
5.  **Edelsbrunner, H., & Harer, J. (2010).** *Computational Topology: An Introduction.* American Mathematical Society.
6.  **AA.VV. (2025).** *LatentFM: A flow-based model operating in the latent space for medical image segmentation.* arXiv:2512.04821.
7.  **Tian, J., et al. (2024).** *Diffuse Attend and Segment: Unsupervised Zero-Shot Segmentation.* CVPR 2024.
8.  **Benamou, J. D., & Brenier, Y. (2000).** *A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem.* Numerische Mathematik.
9.  **Sargsyan, A., et al. (2026).** *FlowDIS: Language-Guided Image Segmentation with Flow Matching.* arXiv:2605.05077.
