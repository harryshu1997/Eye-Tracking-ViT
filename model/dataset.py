# dataset.py
import os, json, glob, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
import h5py

class PersonalGazeDataset(Dataset):
    """
    Your own webcam‐capture dataset.

    Parameters
    ----------
    root_dir : str              # «data/Your_Name»
    screen_w : int              # monitor width  in px
    screen_h : int              # monitor height in px
    transform : torchvision transform | None
    """

    def __init__(self, root_dir, screen_w, screen_h, transform=None):
        self.root_dir   = root_dir
        self.screen_w   = screen_w
        self.screen_h   = screen_h
        self.transform  = transform

        self.items = []   # list of (img_path, (x,y) angle)
        self._index_sessions()

    # ------------------------------------------------------------------ #
    def _index_sessions(self):
        sessions = sorted(glob.glob(os.path.join(self.root_dir, "*")))
        for sess in sessions:
            json_path = os.path.join(sess, "gaze.json")
            if not os.path.isfile(json_path):
                print(f"⚠️  {json_path} missing – skipping session")
                continue
            with open(json_path) as f:
                ann = json.load(f)

            for fname, g in ann.items():
                img_path = os.path.join(sess, fname)
                if not os.path.isfile(img_path):
                    print(f"⚠️  image {img_path} not found – skipped")
                    continue
                # convert screen-pixel coords → (-1,1) angles
                x_ang = (g["x_px"] / self.screen_w) * 2 - 1
                y_ang = (g["y_px"] / self.screen_h) * 2 - 1
                self.items.append((img_path, np.array([x_ang, y_ang],
                                                      dtype=np.float32)))

        print(f"Personal dataset: {len(self.items)} images indexed "
              f"from {len(sessions)} session(s)")

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, gaze = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, torch.tensor(gaze, dtype=torch.float32)
    
# ---------------------------------------------------------------------------
#  Utility: quick file‐structure print-out
# ---------------------------------------------------------------------------
def describe_file(mat_path: str, max_rows: int = 3) -> None:
    """Print shapes / dtypes for one participant file (debug helper)."""
    if not os.path.isfile(mat_path):
        print(f"[inspect] Missing: {mat_path}")
        return
    try:
        with h5py.File(mat_path, "r") as hf:
            if "Data" not in hf or "data" not in hf["Data"] or "label" not in hf["Data"]:
                print(f"[inspect] {mat_path} lacks expected structure.")
                return
            d_img  = hf["Data"]["data"]
            d_lab  = hf["Data"]["label"]
            N      = max(d_lab.shape)
            axis_img = [i for i,s in enumerate(d_img.shape) if s==N][0]
            axis_lab = [i for i,s in enumerate(d_lab.shape) if s==N][0]
            print(f"\n=== {os.path.basename(mat_path)} ===")
            print(f"data  shape : {d_img.shape}, dtype:{d_img.dtype}")
            print(f"label shape : {d_lab.shape}, dtype:{d_lab.dtype}")
            print(f"sample axis in data : {axis_img}")
            print(f"sample axis in label: {axis_lab}")
            print(f"total samples       : {N}")
            preview = np.array(d_lab[:max_rows] if axis_lab==0 else d_lab[:, :max_rows].T)
            print("label preview:\n", preview)
    except Exception as e:
        print(f"[inspect] error reading {mat_path}: {e}")

# ---------------------------------------------------------------------------
#  Main Dataset class
# ---------------------------------------------------------------------------
class MPIFaceGazeDataset(Dataset):
    """
    Memory-efficient loader for the *normalized* MPIIFaceGaze dataset.

    Parameters
    ----------
    root_dir : str
    transform : torchvision transform or None
    use_all_labels : bool
        False → return 2-D gaze only; True → also head-pose + landmarks
    limit_per_participant : int | None
        Random subsample per participant (None = all 3000)
    seed : int | None
        Reproducible subsampling seed
    apply_matlab_fix : bool
        True  → apply RGB→BGR + horizontal flip + 90° CCW rotation
        False → keep images *upright* (default)
    """

    def __init__(self,
                 root_dir: str,
                 transform=None,
                 use_all_labels: bool = False,
                 limit_per_participant: int | None = None,
                 seed: int | None = None,
                 apply_matlab_fix: bool = False):

        self.root_dir         = root_dir
        self.transform        = transform
        self.use_all_labels   = use_all_labels
        self.training         = False      # toggled externally (for jitter)
        self.apply_matlab_fix = apply_matlab_fix

        if seed is not None:
            random.seed(seed)

        self.samples = []
        files_processed = 0

        # ------------ index every participant file -------------------------
        for fname in os.listdir(root_dir):
            if not (fname.startswith("p") and fname.endswith((".mat", ".h5"))):
                continue
            fpath = os.path.join(root_dir, fname)
            pid   = os.path.splitext(fname)[0]

            try:
                with h5py.File(fpath, "r") as hf:
                    g = hf.get("Data")
                    if g is None or "data" not in g or "label" not in g:
                        print(f"⚠️  {fname} skipped: unexpected structure.")
                        continue
                    N = max(g["label"].shape)
                    chosen = (range(N) if limit_per_participant is None or
                              limit_per_participant >= N else
                              random.sample(range(N), limit_per_participant))
                    for i in chosen:
                        self.samples.append(dict(file_path=fpath,
                                                 sample_idx=i,
                                                 participant_id=pid))
                    files_processed += 1
                    print(f"Indexed {pid}: kept {len(chosen)}/{N}")

            except Exception as e:
                print(f"❌  Error indexing {fpath}: {e}")

        random.shuffle(self.samples)
        print(f"Dataset ready: {files_processed} files, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    # ----------------------------------------------------------------------
    def __getitem__(self, idx: int):
        info  = self.samples[idx]
        fpath = info["file_path"]
        sidx  = info["sample_idx"]

        try:
            with h5py.File(fpath, "r") as hf:
                g        = hf["Data"]
                data_d   = g["data"]
                label_d  = g["label"]

                N = max(label_d.shape)
                axis_data  = [i for i,s in enumerate(data_d.shape) if s == N][0]
                axis_label = [i for i,s in enumerate(label_d.shape) if s == N][0]

                # -------- fetch image tensor --------------------------------
                if axis_data == 0:
                    img_np = data_d[sidx, ...]
                elif axis_data == data_d.ndim - 1:
                    img_np = data_d[..., sidx]
                else:
                    img_np = np.take(data_d, sidx, axis=axis_data)

                img_np = np.asarray(img_np).squeeze()
                if img_np.shape[0] == 3:                 # (C,H,W) ➜ (H,W,C)
                    img_np = np.transpose(img_np, (1, 2, 0))

                # -------- optional MATLAB-style orientation fix -------------
                if self.apply_matlab_fix:
                    img_np = img_np[:, :, ::-1]

                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255 if img_np.max() <= 1 else img_np
                              ).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)

                # -------- labels --------------------------------------------
                label_np = (label_d[sidx] if axis_label == 0
                            else label_d[:, sidx]).astype(np.float32)

            gaze = label_np[:2]
            if self.training:
                jitter = 0.05
                gaze += np.random.uniform(-jitter, jitter, 2)
                gaze = np.clip(gaze, -1.0, 1.0)
            gaze_t = torch.tensor(gaze, dtype=torch.float32)

            if self.use_all_labels:
                head = torch.tensor(label_np[2:4],  dtype=torch.float32)
                lmk  = torch.tensor(label_np[4:],   dtype=torch.float32)
                target = {'gaze': gaze_t, 'head_pose': head, 'landmarks': lmk}
            else:
                target = gaze_t

            img = self.transform(img) if self.transform else transforms.ToTensor()(img)
            return img, target

        except Exception as e:
            print(f"❌  Error loading sample {sidx} from {fpath}: {e}")
            dummy = Image.new("RGB", (224, 224), "gray")
            return (self.transform(dummy) if self.transform else transforms.ToTensor()(dummy),
                    torch.zeros(2, dtype=torch.float32))