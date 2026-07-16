# 03. Fondamenti Teorici e Architettura Matematica

Questo documento costituisce la base teorica "unificata" del progetto **FlowStitch**. Unisce l'analisi dell'architettura hardware/software del modello FLUX, la dimostrazione dei limiti dell'inpainting classico e la formulazione matematica delle nuove soluzioni basate sulla Topologia Algebrica e l'alterazione delle equazioni differenziali (ODE). Da questo documento è possibile estrarre interi capitoli per la redazione finale della tesi.

---

## 1. Dal Diffusion al Flow Matching (Il Nuovo Paradigma)

A differenza dei modelli di diffusione tradizionali (es. Stable Diffusion), FLUX.1 abbandona la modellazione esplicita della probabilità basata sul punteggio (Score-Based Models) in favore del **Flow Matching** (o *Rectified Flow*).

Nel Flow Matching, l'obiettivo non è prevedere il rumore aggiunto iterativamente, ma apprendere direttamente un **campo vettoriale** (velocità) che trasporta fluidamente una distribuzione di rumore semplice verso la distribuzione complessa delle immagini naturali. 

Il processo è governato da un'Equazione Differenziale Ordinaria (ODE) che definisce una traiettoria quasi-lineare:
$$ x_t = (1 - t) x_0 + t x_1 $$
Dove:
*   **$x_0$**: Rappresenta il rumore puro iniziale (Oceano Quantistico Gaussiano).
*   **$x_1$**: Rappresenta l'immagine finale nello spazio latente.
*   **$t \in [0, 1]$**: È il tempo del processo di inferenza differenziale.

La rete neurale prevede costantemente la derivata temporale di questa traiettoria, ovvero la velocità $v_t$:
$$ v_t = \frac{dx_t}{dt} = x_1 - x_0 $$

Nella nostra ricerca, il vettore $v_0$ previsto dalla rete allo step inziale codifica la "direzione termodinamica" verso cui l'intero ecosistema spaziale sta puntando. È il nostro atomo di indagine primaria.

---

## 2. L'Architettura MM-DiT e il "Joint Attention"

FLUX.1 adotta un'architettura **MM-DiT** (Multimodal Diffusion Transformer). I classici blocchi Cross-Attention (che mantengono separati testo e immagine) sono sostituiti da blocchi di **Joint Attention**.

L'intuizione alla base è che il testo (prompt) e l'immagine (latenti) debbano interagire profondamente come un'unica sequenza, piuttosto che elaborare l'immagine e condizionarla esternamente. 
Durante questa interazione spaziale, FLUX utilizza i **Rotary Positional Embeddings (RoPE)** per garantire che i token "sappiano dove si trovano" nell'immagine 2D concatenata al testo.

### Come estraiamo i dati (Software Hooking)
Per evitare di alterare il codice nativo o bloccare le performance, utilizziamo un approccio di *Dynamic Class Wrapping* e *Forward Hooking*:
1. Agganciamo un `register_forward_hook` all'intero modulo `Transformer2DModel` per estrarre $x_0$ (input al tempo 0) e $v_0$ (output predittivo al tempo 0).
2. Durante i layer di Joint Attention, effettuiamo un *Monkey-Patching* del `FluxAttnProcessor`. Fermiamo il calcolo iper-ottimizzato di Flash Attention per una frazione di secondo, usiamo un prodotto matriciale esplicito `Softmax(Q K^T)` per materializzare la matrice di attenzione Immagine $\to$ Testo in VRAM, la salviamo e restituiamo il controllo al processore base.
Questo ci permette di estrarre la mappa che ci dice esattamente "quale pixel latente dell'immagine sta guardando quale parola del prompt testuale".

---

## 3. La Crisi delle Maschere Spaziali e il Vuoto Termodinamico

L'approccio storico per l'editing e l'inpainting in Stable Diffusion si basava sul "mascheramento spaziale" (masking statistico) dei tensori. La nostra ricerca ha dimostrato che applicare maschere rigide ai Flow Models è matematicamente scorretto.

### 3.1 Il Vuoto Termodinamico
Tentando di definire maschere basandosi sulla magnitudo dell'energia cinetica $E(x,y) = ||v_{0}(x,y)||_2$, abbiamo scoperto una patologia sistemica.
L'energia cinetica raggiunge i suoi massimi locali ai confini degli oggetti (dove il modello "spinge" per staccare l'oggetto dallo sfondo). Al centro geometrico di un oggetto omogeneo (es. il centro di una sfera liscia), la derivata spaziale è quasi nulla e l'energia crolla verso lo zero per inerzia. 
Applicando filtri e soglie statistiche ($\tau = \mu + \sigma$), il sistema "scava" l'interno dell'oggetto scambiandolo per rumore di fondo, producendo **maschere anulari o frammentate**. Abbiamo ribattezzato questo fenomeno *Vuoto Termodinamico*.

### 3.2 Incompatibilità dei Solver (Ghosting e Discontinuità)
Sovrascrivere una porzione di matrice vettoriale usando una maschera binaria tagliente crea una **Discontinuità di Lipschitz**. Il solver ODE a basso step (es. Eulero) si ritrova a dover gestire salti vettoriali da $100$ a $0$ in un singolo istante di tempo $dt$, il che fa collassare l'equazione differenziale. Il risultato visivo si manifesta con aloni persistenti e blending difettoso.

---

## 4. La Soluzione 1: Topological Data Analysis (TDA)

Per aggirare il Vuoto Termodinamico ed evitare filtri energetici locali, FlowStitch proietta il campo vettoriale e le matrici di attenzione in un framework globale di **Topologia Algebrica** e **Spectral Graph Theory**.

Sostituiamo il banale thresholding con uno *Spectral Co-Embedding Graph Cut*. I pixel diventano nodi di un grafo i cui pesi bilanciano l'affinità di Cross-Attention e la Similarità Coseno del campo di flusso. 
Modellando questo come una nube di punti complessa, calcoliamo l'**Omologia Persistente**:
- **$H_0$ (Componenti Connesse)**: Determina esplicitamente le "isole" semantiche isolate (il target), in base a criteri di nascita (bassa distanza) e morte (coesione).
- **Vettore di Fiedler (Normalized Cut)**: Utilizzando il secondo autovettore più piccolo della matrice Laplaciana del grafo, l'algoritmo spezza nativamente l'immagine lungo i confini fisici "naturali", garantendo l'estrazione non supervisionata dell'oggetto, compresi i suoi vuoti interni.

Il risultato è una maschera continua $A_{target}$ spazialmente perfetta.

---

## 5. La Soluzione 2: Kinetic Trajectory Shaping (KTS)

Una volta isolata la maschera sfumata, come iniettiamo il concetto ($v_{target}$) nel nuovo ambiente ($v_{ambient}$) senza causare la collisione ODE (Ghosting)?
Abbandoniamo la sostituzione spaziale in favore di una **Perturbazione Differenziale nel Dominio del Tempo (ODE Perturbation)**.

### L'Equazione della Perturbazione
Invece di sovrascrivere i tensori, induciamo un campo di forza che "devia" dolcemente la derivata temporale dell'ambiente verso l'attrattore topologico del target:
$$ v_{stitch} = v_{ambient} + \left[ A_{target} \odot (v_{target} - v_{ambient}) \right] \cdot D(t) $$
Dove:
*   $(v_{target} - v_{ambient})$ è il differenziale vettoriale puro, la forza necessaria per passare dallo stato ambientale allo stato target.
*   $A_{target}$ è la densità spaziale estratta topologicamente.
*   $D(t) = e^{-\gamma (t - t_{cutoff})_+}$ è il fattore di **Damping** (Frenata Termodinamica).

### Il Soft-Landing Termodinamico
Il fattore $D(t)$ è cruciale. La letteratura dimostra che un eccesso di energia cinetica per $t \to 1$ causa un sovraccarico in cui il modello si "memorizza", causando bordi seghettati e incoerenza di illuminazione. Smorzando esponenzialmente la nostra interferenza nella parte finale dell'integrazione differenziale (Soft-Landing), costringiamo la rete neurale a dedicare gli ultimi istanti alla stabilizzazione luminosa e al "matching" con lo sfondo ambientale circostante. L'oggetto viene letteralmente "cucito" (Stitched) nel tessuto visivo dell'ecosistema ospite.

---

## 6. Scomposizione di Helmholtz-Hodge (Ricerca Avanzata)

Nel formalismo profondo, il campo vettoriale $v_0$ può essere scomposto secondo il teorema di Helmholtz-Hodge in una componente irrotazionale e una solenoidale:
$$ v_t = \nabla \Phi + \nabla \times \mathbf{A} $$
Il potenziale scalare $\Phi$ domina il puro processo di de-noising strutturale verso distribuzioni naturali, mentre il potenziale vettore $\mathbf{A}$ codifica la **vorticità semantica**, ovvero le perturbazioni locali strettamente necessarie a isolare "quell'oggetto" dallo sfondo universale. Questa scomposizione garantisce che l'estrapolazione di $\tilde{v}_0 = A_{target} \odot v_0$ rappresenti il comando puro per l'oggetto isolato, che può essere salvato nel database locale (Semantic Caching) e richiamato all'infinito senza ricalcolo semantico da zero.
