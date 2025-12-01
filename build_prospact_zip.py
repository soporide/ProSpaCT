
# build_prospact_zip.py
import os
import zipfile


def build_prospact_zip(output_path: str = "prospact_project.zip") -> str:
    """Create a zip file containing all core ProSpaCT project files."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    files_to_zip = [
        "__init__.py",
        "dataset.py",
        "partitioner.py",
        "linear_attention.py",
        "encoders.py",
        "beta_posterior.py",
        "probabilistic_head.py",
        "model.py",
        "losses_metrics.py",
        "train_prospact_toy.py",
        "build_prospact_zip.py",
    ]
    zip_full_path = os.path.join(project_root, output_path)
    with zipfile.ZipFile(zip_full_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in files_to_zip:
            full = os.path.join(project_root, fname)
            if os.path.exists(full):
                zf.write(full, arcname=fname)
    return zip_full_path


if __name__ == "__main__":
    path = build_prospact_zip()
    print(f"Created zip at: {path}")
