# FlowStitch: Manuale Operativo e Architetturale (User Guide)

Benvenuto nella nuova architettura **FlowStitch**. Questo documento serve a orientarti dopo il massiccio refactoring che ha trasformato i tuoi notebook sperimentali in un vero e proprio pacchetto Python formale, modulare e pronto per la ricerca di livello accademico.

---

## 1. Cosa è cambiato? (Dal Caos al Framework)

Precedentemente, la logica di estrazione delle maschere e modifica dei tensori era sparsa in script disordinati (es. `local_analysis`). Ogni volta che volevi testare qualcosa, dovevi caricare gigabyte di pesi in VRAM e rischiavi di rompere la logica pregressa.

Ora il codice vive dentro la cartella `flowstitch/`, organizzato per domini di competenza. **Non modificherai più codice copiando/incollando celle**, ma importerai funzioni pulite (es. `from flowstitch.extraction.spectral_mask import compute_fiedler_mask`).

### La struttura del progetto:
- `flowstitch/core/`: Il "motore". Contiene `flux_hooking.py`, che gestisce fisicamente l'intercettazione dei tensori dal modello Flux senza romperlo.
- `flowstitch/extraction/`: La "vista". Contiene `spectral_mask.py`, il modulo matematico che trasforma le mappe di attenzione in grafi (DiffCut) per ricavare i bordi esatti degli oggetti.
- `flowstitch/stitching/`: Il "bisturi". Contiene `kts.py` ed `ema_smoothing.py` per manipolare il campo vettoriale (ODE) in modo termodinamicamente stabile.
- `flowstitch/pipelines/`: Gli script "assemblatori" che uniscono Core, Extraction e Stitching.
- `hpc_cluster/`: Gli script pronti da lanciare sulle macchine con GPU pesanti.
- `notebooks/`: L'ambiente interattivo (come `00_master_validation_pipeline.ipynb`) che funge da "lavagna" per testare concettualmente la pipeline passo-passo.

---

## 2. Il Cuore Fisico: Cosa estraiamo e cosa iniettiamo?

Il tuo modello di riferimento è **Flux.1 (MMDiT)**. Essendo un Transformer Multimodale colossale, non ha le classiche "U-Net". 

### A) Cosa Estraiamo (Hooking Passivo)
Usando il `FluxDataCapturer` (in `core/flux_hooking.py`), noi eseguiamo un "intervento chirurgico a cuore aperto" durante la generazione dell'immagine.
Quando Flux tenta di calcolare la *Full Attention* (nei layer a flusso singolo, es. dal layer 19 in poi), noi la intercettiamo:
1. **$v_0$ (Velocity Vector)**: Catturiamo il vettore velocità iniziale della ODE.
2. **$x_0$ (Noise Vector)**: Catturiamo il rumore di partenza.
3. **Cross-Attention Maps**: Catturiamo i tensori di attenzione *tra i token di testo e i token immagine*. 

> [!TIP]
> **Perché lo facciamo?** Perché in queste mappe si nasconde la "segmentazione zero-shot". Il modello sa esattamente dove si trova "la mela rossa" a livello spaziale prima ancora di averla disegnata.

### B) Cosa Iniettiamo/Modifichiamo (Intervento Attivo)
Una volta estratta l'attenzione, entra in gioco la manipolazione:
1. **Spectral Matting & TDA**: Non usiamo più semplici binarizzazioni. Calcoliamo un autovettore Laplaciano (Fiedler Vector) oppure applichiamo la **Topological Data Analysis (TDA)** (`extract_tda_mask` in `flowstitch.extraction.tda_mask`) per raggruppare i vettori latenti di velocità tramite single-linkage clustering (omologia H0) e risolvere il "Vuoto Termodinamico".
2. **KTS (Kinetic Trajectory Shaping)**: Iniettiamo un freno termodinamico ($D(t)$) alla ODE per prevenire la singolarità terminale per $t \to 1$ ed evitare la saturazione dei bordi con rumore ad alta frequenza.
3. **Look-Back EMA**: Eseguiamo uno smoothing temporale integrato direttamente nel ciclo ODE (`perform_ode_step` con parametro `smoother`) aggiornando la media mobile della velocità per attenuare le oscillazioni repentine dei token.

---

## 3. Ambiente Locale (MacBook) vs Ambiente HPC (Cluster)

Questa è la distinzione più importante per il tuo flusso di lavoro quotidiano. Flux.1-dev richiede tra i **16 e i 24+ GB di VRAM** per essere eseguito integralmente. 

### Cosa PUOI e DEVI fare in Locale (sul tuo Mac):
- **Scrivere codice e testare la logica**: Modificare le formule in `spectral_mask.py` o `kts.py`.
- **Eseguire test "Mock"**: Lanciare il notebook `00_master_validation_pipeline.ipynb`. Ho inserito dei blocchi `try/except` apposta: il tuo Mac non esploderà; il notebook fallirà silenziosamente il caricamento della GPU, ma andrà avanti a validare l'importazione delle librerie e il flusso delle variabili mockate.
- **Calcolare le Matrici e Grafi offline**: Se hai salvato un file `.pt` (PyTorch Tensor) scaricato dal cluster, puoi elaborarne il *Normalized Cut* (Laplaciano) sul tuo Mac. Le operazioni matriciali matematiche girano sulla CPU del Mac in pochi secondi.
- **Scrivere la tesi e documentazione**.

### Cosa NON PUOI fare in Locale (riservato al Cluster GPU):
- **Generare Immagini reali da prompt**: L'inizializzazione di `FluxPipeline` su un Mac standard causerà un crash per *Out Of Memory*.
- **Estrarre Attenzione in tempo reale**: L'hooking vero e proprio richiede che la rete passi forward/backward. 

> [!WARNING]
> Mai lanciare `python hpc_cluster/run_dataset.py` sul tuo MacBook. Congeleresti il sistema operativo per saturazione della memoria unificata (SWAP).

---

## 4. Flusso di Lavoro (Come procedere d'ora in poi)

1. **Ideazione**: Apri `00_master_validation_pipeline.ipynb` in locale per capire i passaggi. Se ti viene una nuova idea matematica (es. un nuovo calcolo per smorzare l'ODE), scrivi la funzione in `flowstitch/stitching/`.
2. **Deploy sul Cluster**: Clona o sincronizza il progetto sul nodo HPC.
3. **Cattura Dati Massiva**: Lancia `python hpc_cluster/run_dataset.py`. Questo script girerà ore e sputerà fuori gigabyte di file (`v0_velocity.pt`, `attention_maps.pt`) dentro la cartella `data/`.
4. **Analisi/Stitching**: Prendi quei file ed elaborali con `python hpc_cluster/run_stitching.py` (sul cluster o anche sul Mac se sposti i file localmente). Questo applicherà il KTS e l'EMA e ti restituirà i DICE/IoU score e le immagini modificate.
5. **Redazione Tesi**: Riporta i DICE score nella tesi in `docs/Tesi_FlowStitch_Avanzata.tex`.

Sei ora in possesso di un'architettura che implementa lo "stato dell'arte" decritto dai paper del 2025/2026. Buon lavoro!
