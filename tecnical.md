# Analisi Architetturale di FLUX.1 e Metodologia di Estrazione Dati

Questo documento illustra nel dettaglio l'architettura del modello **FLUX.1** (sviluppato da Black Forest Labs), il paradigma matematico alla base del suo funzionamento, e la metodologia di *software hooking* implementata per l'estrazione a basso livello delle variabili fisiche e delle mappe di attenzione. Queste metriche costituiscono il dataset primario per l'analisi della "Planimetria Architettonica" del processo generativo.

---

## 1. Fondamenti Matematici: Dal Diffusion al Flow Matching

A differenza dei modelli di diffusione tradizionali (es. Stable Diffusion), FLUX.1 abbandona la modellazione esplicita della probabilità basata sul punteggio (Score-Based Models) in favore del **Flow Matching** (o *Rectified Flow*).

### Il Modello Generativo
Nel Flow Matching, l'obiettivo del modello non è prevedere il rumore aggiunto, ma apprendere direttamente un campo vettoriale (velocità) che trasporta fluidamente una distribuzione di rumore semplice verso la distribuzione complessa dei dati (immagini reali). 

Il processo è governato da una Equazione Differenziale Ordinaria (ODE) che definisce un percorso lineare (traiettoria retta):
$$ x_t = (1 - t) x_0 + t x_1 $$
Dove:
*   **$x_0$**: Rappresenta il rumore puro iniziale (Campionato da una gaussiana).
*   **$x_1$**: Rappresenta l'immagine finale (spazio latente).
*   **$t \in [0, 1]$**: È il tempo del processo di inferenza.

L'obiettivo della rete neurale (il Transformer) è prevedere la derivata temporale di questa traiettoria, ovvero il vettore velocità **$v_t$**:
$$ v_t = \frac{dx_t}{dt} = x_1 - x_0 $$

Nella nostra ricerca, l'estrazione allo step iniziale ($t=0$) è critica, in quanto il vettore $v_0$ previsto dalla rete contiene la "direzione iniziale" verso cui l'intero processo generativo sta puntando, formando la base del blueprint architetturale.

---

## 2. Architettura Multimodale: MM-DiT (Multimodal Diffusion Transformer)

FLUX.1 adotta un'architettura **MM-DiT**, in cui i classici blocchi Cross-Attention (che mantengono separati testo e immagine) sono sostituiti da blocchi di **Joint Attention**.

L'intuizione alla base del MM-DiT è che il testo (prompt) e l'immagine (latenti) debbano interagire profondamente e simmetricamente come un'unica sequenza ininterrotta, piuttosto che elaborare l'immagine e condizionarla esternamente col testo.

### Il Flusso dei Dati
1.  **Testo (`encoder_hidden_states`)**: Viene codificato dai text encoder (es. T5-XXL) producendo una sequenza di embedding testuali di dimensione `[batch, n_text_tokens, dim]`.
2.  **Immagine (`hidden_states`)**: L'input di rumore latente viene scomposto in patch (simile al Vision Transformer) formando una sequenza `[batch, n_image_tokens, dim]`.

Invece di elaborare i due flussi in parallelo, i *Joint Transformer Blocks* di FLUX.1 uniscono testo e immagine concatenandoli lungo la dimensione della sequenza.

---

## 3. Il Meccanismo di Joint Attention (Sotto il cofano)

Nel modulo di Joint Attention intervengono svariati strati di algebra tensoriale specifici che ci hanno costretto ad un intervento chirurgico durante l'implementazione dell'estrattore.

L'algoritmo di attenzione originale di FLUX procede in questi step:
1.  **Proiezioni Indipendenti**: Le due modalità vengono proiettate nei rispettivi Query ($Q$), Key ($K$) e Value ($V$).
    *   Immagine: `to_q(img)`, `to_k(img)`
    *   Testo: `add_q_proj(txt)`, `add_k_proj(txt)`
2.  **Suddivisione per Teste (Heads)**: I tensori vengono rimodellati dividendo la dimensione nascosta (es. 3072) nel numero di teste di attenzione (es. 24 teste da 128 dimensioni ciascuna).
3.  **RMSNorm (Root Mean Square Normalization)**: A differenza di molti modelli che normalizzano prima della proiezione, FLUX applica la RMSNorm *direttamente sulle proiezioni Query e Key* (ma solo per alcune componenti specifiche) al fine di stabilizzare il gradiente nel training ad alta risoluzione.
4.  **Concatenazione**: I tensori Query e Key di testo e immagine vengono uniti in una singola sequenza.
5.  **Rotary Positional Embeddings (RoPE)**: Poiché la sequenza concatenata perde intrinsecamente le nozioni spaziali bidimensionali dell'immagine, FLUX applica i RoPE direttamente alle Query e alle Key prima di calcolare i pesi. Questo garantisce che i token sappiano "dove si trovano" nell'immagine 2D.
6.  **Scaled Dot-Product Attention**: Calcolo classico dell'attenzione $Softmax\left(\frac{Q K^T}{\sqrt{d_k}}\right)$.

---

## 4. Metodologia di Intervento e Intercettazione (Il nostro Software)

Per prelevare i dati necessari alla ricerca ($x_0$, $v_0$ e mappe di attenzione) senza intaccare o ritardare il normale funzionamento di FLUX, abbiamo sviluppato una logica non intrusiva basata su **Hooking in PyTorch** e **Dynamic Class Wrapping**.

### A. Estrazione Cinematica: $x_0$ e $v_0$
Abbiamo agganciato un `register_forward_hook` all'intero modulo `Transformer2DModel`.
Questo hook ci permette di "spiare" i tensori in entrata (`input`) e in uscita (`output`) al momento del calcolo.
Poiché la libreria `diffusers` passa l'input come *Keyword Argument* (`hidden_states`), abbiamo configurato l'hook con `with_kwargs=True` per decodificare il parametro nominale.
*   **$x_0$**: Viene catturato come l'`hidden_states` d'ingresso al primissimo step del sampler (step 0).
*   **$v_0$**: Corrisponde all'output del modello allo step 0, ovvero la derivata predittiva.
*   **$x_{pred}$**: Calcolato deterministicamente sommando $x_0 + v_0$, che indica dove la rete sta "guardando" per il risultato finale.

### B. Estrazione della Cross-Attention Immagine-Testo
Il calcolo interno dell'attenzione in `diffusers` è isolato in oggetti opachi di tipo `Processor` (nello specifico, `FluxAttnProcessor2_0`). Questi processori sono progettati per l'efficienza (usando Flash Attention e omettendo il ritorno delle matrici dei pesi).

Per catturare queste matrici abbiamo agito così:
1.  **Monkey-Patching Dinamico**: Durante l'esecuzione, lo script sostituisce temporaneamente il processor originale (`attn.processor`) dei layer d'interesse (es. Layer 0 e Layer 10) con una nostra classe personalizzata `AttnProcessorWrapper`.
2.  **Calcolo Algebrico Parallelo**: Il nostro wrapper emula matematicamente i primi passi del processore originale. Esegue le proiezioni, la suddivisione per `heads`, l'RMSNorm e la concatenazione.
3.  **Applicazione del RoPE**: Intercettiamo il tensore `image_rotary_emb` e lo applichiamo in tempo reale alle nostre query e key clonate usando l'algoritmo nativo di diffusers (`apply_rope`).
4.  **Estrazione del Blocco di Interesse**: Invece di calcolare l'efficienza opaca di FlashAttention, eseguiamo una moltiplicazione matriciale pura (`torch.matmul`) e applichiamo Softmax per materializzare i pesi in memoria RAM.
    Poiché la matrice risultante è una enorme Joint Attention (Testo->Testo, Testo->Immagine, Immagine->Testo, Immagine->Immagine), filtriamo esclusivamente il quadrante **Immagine $\to$ Testo**:
    ```python
    cross_attn = attn_weights[:, :, n_text:, :n_text]
    ```
    Questa matrice ci dice esattamente "quale pixel latente dell'immagine sta guardando quale specifica parola del prompt testuale".
5.  **Passaggio Trasparente**: Subito dopo l'estrazione della mappa (con gradienti disabilitati tramite `torch.no_grad` per non far esplodere la memoria VRAM), il wrapper passa la palla al processore di base altamente ottimizzato per calcolare il vero forward, non influenzando minimamente le performance della generazione.
6.  **Pulizia**: Al termine della generazione del singolo prompt, tutte le spie vengono smontate e il modello viene riportato al suo stato originale intonso.
