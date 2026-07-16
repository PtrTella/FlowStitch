# 02. Guida Utente e Flusso di Lavoro (Pipeline)

Questo documento costituisce il manuale operativo per utilizzare l'architettura **FlowStitch**, pensata per operare sia in locale (analisi teorica) che su cluster HPC (estrazione massiva).

## 1. Architettura del Software
Il codice sorgente è organizzato modularmente nella cartella `flowstitch/` per evitare la frammentazione tipica dei notebook sperimentali. Le funzioni sono importabili e strutturate per domini di competenza:

* **`flowstitch/core/`**: Il "motore". Contiene `flux_hooking.py`, che gestisce fisicamente l'intercettazione dei tensori dal modello FLUX senza interrompere il suo normale forward pass.
* **`flowstitch/extraction/`**: La "vista". Contiene i moduli matematici (`spectral_mask.py`, `tda_mask.py`) che trasformano le mappe di attenzione in grafi (DiffCut) o complessi simpliciali per ricavare i bordi esatti degli oggetti.
* **`flowstitch/stitching/`**: Il "bisturi". Contiene `kts.py` e `ema_smoothing.py` per manipolare il campo vettoriale (ODE) in modo termodinamicamente stabile durante l'integrazione differenziale.
* **`flowstitch/pipelines/`**: Gli script ad alto livello che orchestrano il flusso di dati (generazione -> estrazione -> iniezione).

Inoltre:
* **`hpc_cluster/`**: Script SLURM e file batch ottimizzati per le macchine remote con GPU pesanti.
* **`notebooks/`**: L'ambiente interattivo (es. `00_master_validation_pipeline.ipynb`) che funge da "lavagna" sicura per testare concettualmente la pipeline logica senza far esplodere la memoria locale.

## 2. Il Cuore Fisico: Cosa estraiamo e cosa iniettiamo?
Il modello di riferimento è **FLUX.1 (MM-DiT)**. Essendo un Transformer Multimodale colossale, non presenta le classiche U-Net.

### A) Cosa Estraiamo (Hooking Passivo)
Usando il `FluxDataCapturer`, eseguiamo un "intervento chirurgico a cuore aperto" durante la generazione dell'immagine. Intercettiamo il momento in cui FLUX calcola la *Full Attention* (nei layer a flusso singolo):
1. **$v_0$ (Velocity Vector)**: Il vettore velocità iniziale della ODE.
2. **$x_0$ (Noise Vector)**: L'oceano di rumore di partenza.
3. **Cross-Attention Maps**: I tensori di attenzione *tra i token di testo e i token immagine*. In queste mappe risiede la "segmentazione zero-shot" non supervisionata.

### B) Cosa Iniettiamo/Modifichiamo (Intervento Attivo)
Una volta estratta la logica spaziale dell'oggetto, entra in gioco la manipolazione dell'equazione differenziale (ODE):
1. **Spectral Matting & TDA**: Calcoliamo un autovettore Laplaciano (Fiedler Vector) o usiamo la Topological Data Analysis per raggruppare i vettori di velocità e generare una maschera continua, risolvendo il problema del "Vuoto Termodinamico".
2. **KTS (Kinetic Trajectory Shaping)**: Iniettiamo un freno termodinamico ($D(t)$) alla ODE. Questo impedisce le singolarità terminali per $t \to 1$, evitando "l'incidente" del solver differenziale sui contorni.
3. **Look-Back EMA**: Eseguiamo uno smoothing temporale integrato direttamente nel ciclo ODE, aggiornando la media mobile della velocità per attenuare salti repentini di traiettoria visiva.

## 3. Ambiente Locale (MacBook) vs Ambiente HPC (Cluster Giano)

FLUX.1 richiede tra i **16 e i 24+ GB di VRAM** per un'esecuzione integrale. 

### Operazioni consentite in Locale (MacBook):
* Scrivere codice e testare la logica formale (es. aggiornare formule matematiche).
* Eseguire test "Mock" in `00_master_validation_pipeline.ipynb` (che simula l'ambiente senza caricare pesi massicci in memoria).
* Calcolare le Matrici e i Grafi offline su file `.pt` pre-estratti scaricati dal cluster.
* Scrittura teorica e analisi.

### Operazioni riservate all'HPC (Cluster):
* Generare immagini reali da prompt.
* Estrarre mappe di attenzione in tempo reale (hooking attivo).
* **Mai** lanciare run complete (`run_dataset.py`) localmente, pena il blocco del sistema operativo per saturazione SWAP.

## 4. Flusso di Lavoro Ottimale
Per portare a termine l'esperimento:
1. **Sviluppo:** Testa la logica matematica e l'assemblaggio dei moduli sui notebook locali.
2. **Deploy:** Sincronizza il progetto sul nodo HPC. Usa `pip install -e ".[hpc]" --no-cache-dir` nel nodo interattivo.
3. **Estrazione Massiva:** Lancia `run_dataset.sbatch`. Lo script estrarrà gigabyte di file tensoriali (`v0_velocity.pt`, `attention_maps.pt`) nella cartella `data/`. Questo è il tuo "Database Latente" (Semantic Cache).
4. **Stitching e Analisi:** Elabora i dati salvati applicando KTS tramite `run_experiment.sbatch`. Otterrai le immagini finali combinate e i punteggi quantitativi (DICE/IoU).
5. **Reportistica:** Riporta i risultati e i DICE score direttamente nella tesi LaTeX.
