import pandas as pd
import os

def download_nasa_datasets(cache_dir="datasets"):
    """
    Download NASA exoplanet datasets or load from local cache if available.
    """

    os.makedirs(cache_dir, exist_ok=True)

    datasets = {
        "koi": ("Kepler KOI - Cumulative table",
                "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"),
        "toi": ("TESS TOI table",
                "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"),
        "k2": ("K2 Planets and Candidates",
               "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=csv")
    }

    results = {}

    for key, (name, url) in datasets.items():
        cache_path = os.path.join(cache_dir, f"{key}.csv")

        if os.path.exists(cache_path):
            print(f"\n📂 Loading {name} from cache...")
            df = pd.read_csv(cache_path, low_memory=False)
        else:
            print(f"\n🔄 Downloading {name}...")
            df = pd.read_csv(url, low_memory=False)
            df.to_csv(cache_path, index=False)
            print(f"💾 Saved {name} to {cache_path}")

        print(f"✅ Finished {name}, shape = {df.shape}")
        print(df.head(3))
        results[key] = df

    return results


# Load datasets (from web the first time, then cache)
datasets = download_nasa_datasets()

# Example usage
koi_data = datasets["koi"]
toi_data = datasets["toi"]
k2_data = datasets["k2"]
