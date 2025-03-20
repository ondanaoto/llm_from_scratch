import subprocess
from pathlib import Path


def main():
    base_dir = Path(__file__).parent

    scripts = ["data_prepare.py", "ftdata_preprocess.py"]

    for script in scripts:
        script_path = base_dir / script
        print(f"Running {script_path.name}...")
        result = subprocess.run(["python", str(script_path)], check=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"{script_path.name} failed with return code {result.returncode}"
            )

    print("Initialization completed successfully.")


if __name__ == "__main__":
    main()
