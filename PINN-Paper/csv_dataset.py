import torch
from torch.utils.data import Dataset, DataLoader, Subset # Assicurati che Subset sia importato
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

class QuadrotorDataset(Dataset):
    """
    Dataset per dati di quadrirotore, caricando stati e input da file CSV separati.
    Prepara i dati per essere usati con un modello e una loss fisica,
    gestendo la separazione di dati scalati e non scalati.
    """
    def __init__(self, state_folder, input_folder):
        """
        Inizializza il dataset caricando e pre-processando i dati.

        Args:
            state_folder (str): Path alla cartella contenente i file CSV degli stati.
                                Formato atteso per file stato: (t, x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz)
            input_folder (str): Path alla cartella contenente i file CSV degli input.
                                Formato atteso per file input: (t, thrust, tau_phi, tau_theta, tau_psi)
        """
        self.state_folder = state_folder
        self.input_folder = input_folder

        # Trova e ordina i file per assicurare corrispondenza
        try:
            self.state_files = sorted([os.path.join(state_folder, f) for f in os.listdir(state_folder) if f.endswith('.csv')])
            self.input_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Errore nel trovare le cartelle: {e}. Verifica i path: State='{state_folder}', Input='{input_folder}'") from e

        if not self.state_files or not self.input_files:
            print(f"Warning: Nessun file CSV trovato in una o entrambe le cartelle: State='{state_folder}', Input='{input_folder}'")
            # Potresti voler generare un errore qui se non è previsto che siano vuote
            # raise ValueError("Nessun file CSV trovato nelle cartelle specificate.")

        if len(self.state_files) != len(self.input_files):
            print(f"Warning: Numero di file non corrispondente! State: {len(self.state_files)}, Input: {len(self.input_files)}. Verranno usate solo le coppie corrispondenti per nome (dopo l'ordinamento).")
            # Potrebbe essere necessaria una logica più robusta per accoppiare i file se i nomi non bastano

        # --- CARICA DATI ORIGINALI ---
        all_states_list = []
        all_inputs_list = []
        print(f"Loading data files...")
        loaded_pairs = 0
        skipped_pairs = 0
        for state_file, input_file in zip(self.state_files, self.input_files):
            # Verifica base sulla corrispondenza dei nomi file (opzionale, dipende dalla tua convenzione)
            # state_basename = os.path.splitext(os.path.basename(state_file))[0]
            # input_basename = os.path.splitext(os.path.basename(input_file))[0]
            # if state_basename != input_basename:
            #     print(f"Warning: Skipped non-matching file names: {state_file}, {input_file}")
            #     skipped_pairs += 1
            #     continue

            try:
                # Carica TUTTE le colonne (incluso tempo)
                state_data = pd.read_csv(state_file, header=None).values
                input_data = pd.read_csv(input_file, header=None).values

                # Controlli di validità
                if state_data.shape[0] == 0 or input_data.shape[0] == 0:
                    print(f"Warning: Empty file content in {state_file} or {input_file}. Skipping pair.")
                    skipped_pairs += 1
                    continue
                if state_data.shape[0] != input_data.shape[0]:
                    print(f"Warning: Mismatch length {len(state_data)} vs {len(input_data)} in {state_file}, {input_file}. Skipping pair.")
                    skipped_pairs += 1
                    continue
                if state_data.shape[1] != 13:
                     print(f"Warning: Unexpected number of columns in state file {state_file} (Expected 13, Got {state_data.shape[1]}). Skipping pair.")
                     skipped_pairs += 1
                     continue
                if input_data.shape[1] != 5:
                     print(f"Warning: Unexpected number of columns in input file {input_file} (Expected 5, Got {input_data.shape[1]}). Skipping pair.")
                     skipped_pairs += 1
                     continue

                all_states_list.append(state_data)
                all_inputs_list.append(input_data)
                loaded_pairs += 1
            except pd.errors.EmptyDataError:
                 print(f"Warning: EmptyDataError for {state_file} or {input_file}. Skipping.")
                 skipped_pairs += 1
                 continue
            except Exception as e:
                 print(f"Error processing file pair {state_file}, {input_file}: {e}. Skipping.")
                 skipped_pairs += 1
                 continue

        if not all_states_list:
             raise ValueError("No valid data loaded after checking files. Check paths, file contents, and formats.")
        print(f"Successfully loaded {loaded_pairs} file pairs. Skipped {skipped_pairs} pairs.")

        # Concatena tutti i dati originali
        # Assume che la concatenazione mantenga l'ordine temporale se i file sono ordinati
        self.original_states = np.concatenate(all_states_list, axis=0) # (N, 13) -> t, x...wz
        self.original_inputs = np.concatenate(all_inputs_list, axis=0) # (N, 5)  -> t, thrust, taux, tauy, tauz
        print(f"Total data points concatenated: {len(self.original_states)}")


        # --- PREPARA DATI PER MODELLO E MSE (SENZA TEMPO) ---
        # Stati per MSE (x..wz) -> verranno scalati
        self.states_for_scaling = torch.tensor(self.original_states[:, 1:], dtype=torch.float32) # Shape (N, 12)
        # Input per modello (thrust, taux..tauz) -> verranno scalati
        self.inputs_for_scaling = torch.tensor(self.original_inputs[:, 1:], dtype=torch.float32) # Shape (N, 4)

        # --- PREPARA DATI PER PHYSICS LOSS (NON SCALATI) ---
        self.times = torch.tensor(self.original_states[:, 0], dtype=torch.float32)              # Shape (N,) Tempo
        self.torques = torch.tensor(self.original_inputs[:, 2:], dtype=torch.float32)           # Shape (N, 3) Coppie tau_x, tau_y, tau_z (NON SCALATE)

        # Placeholder per gli scaler e i dati scalati; verranno popolati da create_dataloaders
        self.state_scaler = None
        self.input_scaler = None
        self.scaled_states = None
        self.scaled_inputs = None

    def __len__(self):
        """Restituisce il numero totale di campioni nel dataset."""
        return len(self.original_states)

    def __getitem__(self, idx):
        """
        Restituisce un singolo campione dal dataset.

        Args:
            idx (int): Indice del campione richiesto.

        Returns:
            tuple: (model_input, state_target, physics_info)
                - model_input (Tensor): Input scalato per il modello (thrust, taux, tauy, tauz) + rumore. Shape (4,).
                - state_target (Tensor): Stato target scalato per la loss MSE (x..wz). Shape (12,).
                - physics_info (Tensor): Informazioni non scalate per la loss fisica (tempo, taux, tauy, tauz). Shape (4,).
        """
        # Controlla se i dati sono stati scalati (dovrebbe essere fatto da create_dataloaders)
        if self.scaled_inputs is None or self.scaled_states is None:
             raise RuntimeError("Dataset non è stato scalato. Chiamare prima create_dataloaders.")

        # 1. Input per il modello (scalato) + rumore
        scaled_input = self.scaled_inputs[idx]
        # Considera se applicare rumore solo nel dataloader di training
        noise = torch.randn_like(scaled_input) * 0.02 # Aggiunge 2% di rumore gaussiano
        model_input = scaled_input + noise

        # 2. Target per MSE (stato scalato)
        state_target = self.scaled_states[idx]

        # 3. Info per Physics Loss (tempo e coppie non scalate)
        # Concatena il tempo (scalare reso 1D) con le coppie (già 1D di shape 3)
        physics_info = torch.cat((self.times[idx].unsqueeze(0), self.torques[idx])) # Risultato shape (1+3=4,)

        return model_input, state_target, physics_info

# --- FUNZIONE create_dataloaders MODIFICATA PER SPLIT SEQUENZIALE ---
def create_dataloaders(dataset, batch_size=128, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Crea dataloader con SPLIT SEQUENZIALE del dataset e SCALA i dati.
    Lo split sequenziale è necessario per rendere valide le differenze finite
    nella loss fisica quando si usa shuffle=False nel DataLoader di training.

    Args:
        dataset (QuadrotorDataset): Istanza di QuadrotorDataset (dati non ancora scalati).
        batch_size (int): Dimensione del batch per i DataLoader.
        train_ratio (float): Rapporto di dati da usare per il training (es. 0.7 per 70%).
        val_ratio (float): Rapporto di dati da usare per la validazione (es. 0.15 per 15%).
                           Il test set userà il rimanente (1 - train_ratio - val_ratio).
        seed (int): Seed per la generazione di numeri casuali (per riproducibilità, anche se lo split è deterministico).

    Returns:
        tuple: (train_loader, val_loader, test_loader, state_scaler, input_scaler)
            - train_loader (DataLoader): DataLoader per il training set (sequenziale, no shuffle).
            - val_loader (DataLoader): DataLoader per il validation set (sequenziale, no shuffle).
            - test_loader (DataLoader): DataLoader per il test set (sequenziale, no shuffle).
            - state_scaler (StandardScaler): Scaler fittato sugli stati del training set.
            - input_scaler (StandardScaler): Scaler fittato sugli input del training set.
    """
    # Imposta il seed per coerenza (anche se non c'è casualità nello split)
    torch.manual_seed(seed)
    np.random.seed(seed) # Anche per numpy se usato implicitamente

    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Il dataset è vuoto, impossibile creare i DataLoader.")

    # Calcola le dimensioni degli split
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    # Assicura che train_size e val_size non siano negativi o zero se possibile
    train_size = max(0, train_size)
    val_size = max(0, val_size)

    test_size = dataset_size - train_size - val_size
    # Gestisci il caso in cui test_size diventa negativo a causa degli arrotondamenti
    if test_size < 0:
        print(f"Warning: Negative test_size ({test_size}) calculated due to rounding with small dataset size. Adjusting validation size.")
        val_size += test_size # Riduci la dimensione della validazione per compensare
        val_size = max(0, val_size) # Assicura che non sia negativo
        test_size = 0 # Imposta test_size a 0
        # Ricalcola train_size se necessario per assicurare che la somma sia dataset_size
        if train_size + val_size > dataset_size:
             train_size = dataset_size - val_size
             train_size = max(0, train_size)

    print(f"Dataset size: {dataset_size}")
    print(f"Creating sequential split: Train={train_size}, Val={val_size}, Test={test_size}")

    if train_size == 0:
        raise ValueError("Training split size is 0. Cannot train or fit scalers. Check ratios/dataset size.")

    # --- CREA INDICI SEQUENZIALI ---
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, dataset_size))

    # Estrai dati di training (NON SCALATI) usando gli indici SEQUENZIALI per fittare gli scaler
    print("Fitting scalers on the sequential training split...")
    train_states_unscaled = dataset.states_for_scaling[train_indices]
    train_inputs_unscaled = dataset.inputs_for_scaling[train_indices]

    # Inizializza e fitta gli scaler SOLO sui dati di training
    state_scaler = StandardScaler()
    input_scaler = StandardScaler()
    state_scaler.fit(train_states_unscaled.numpy())
    input_scaler.fit(train_inputs_unscaled.numpy())
    print("Scalers fitted.")

    # Applica la normalizzazione all'INTERO dataset (usando gli scaler fittati)
    # Questo è fatto per semplicità, così __getitem__ accede sempre a dati scalati
    print("Applying scaling to the entire dataset...")
    dataset.scaled_states = torch.tensor(state_scaler.transform(dataset.states_for_scaling.numpy()), dtype=torch.float32)
    dataset.scaled_inputs = torch.tensor(input_scaler.transform(dataset.inputs_for_scaling.numpy()), dtype=torch.float32)
    print("Scaling applied.")


    # Salva gli scaler nell'oggetto dataset (può essere utile)
    dataset.state_scaler = state_scaler
    dataset.input_scaler = input_scaler

    # Crea i Subset Pytorch usando gli indici SEQUENZIALI
    train_subset = Subset(dataset, train_indices)
    # Crea Subset vuoti se le dimensioni sono 0
    val_subset = Subset(dataset, val_indices) if val_size > 0 else None
    test_subset = Subset(dataset, test_indices) if test_size > 0 else None

    # Crea i DataLoader usando i Subset SEQUENZIALI
    # IMPORTANTE: Usa shuffle=False anche per il training loader!
    #             Usa drop_last=True per train/val per avere batch consistenti per le diff. finite.
    print("Creating DataLoaders (shuffle=False, drop_last=True for train/val)...")
    num_workers = 0 # Imposta a 0 per debug, aumenta (es. 2 o 4) per performance se non causa problemi
    pin_memory = torch.cuda.is_available() # Usa pin_memory solo se hai una GPU

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False, # NECESSARIO per differenze finite valide
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True # Assicura batch di dimensione costante per diff. finite
    )

    # Crea val_loader solo se val_subset esiste
    val_loader = None
    if val_subset:
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True # Coerenza con train_loader
        )

    # Crea test_loader solo se test_subset esiste
    test_loader = None
    if test_subset:
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False # Valuta tutti i campioni nel test set
        )

    print(f"Data scaled. Scaled states shape: {dataset.scaled_states.shape}, Scaled inputs shape: {dataset.scaled_inputs.shape}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}, Test batches: {len(test_loader) if test_loader else 0}")


    return train_loader, val_loader, test_loader, state_scaler, input_scaler