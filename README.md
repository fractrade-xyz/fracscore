# FRAC Loyalty Score Script

This script calculates and retrieves the **FRAC Loyalty Score** for a given wallet address using cached or live data from the Hypurrscan API.


## **1. Installation & Setup**
### **Step 1: Install Poetry (if not installed)**
If you don't have **Poetry**, install it with:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Alternatively, use **pip**:
```bash
pip install poetry
```
Check if it's installed correctly:
```bash
poetry --version
```

### **Step 2: Clone the Repository**
```bash
git clone https://github.com/fractrade-xyz/fracscore.git
cd fracscore
```

### **Step 3: Install Dependencies & Create a Virtual Environment**
Poetry automatically manages virtual environments. Run:
```bash
poetry install
```
This will:
- Create a **virtual environment**.
- Install all dependencies.

### **Step 4: Activate the Virtual Environment**
Poetry manages environments, so you can **run commands directly** using:
```bash
poetry run python fracscore.py --help
```
or enter the shell:
```bash
poetry shell
```

## **2. Usage**
### **Check FRAC Loyalty Score for a Wallet**
Run:
```bash
poetry run python fracscore.py 0xYourWalletAddress
```
Example:
```bash
poetry run python fracscore.py 0xbf8802002d394f6f53d1c5a5cc500fd8dfa3b7c8
```
**Output Example:**
```
Wallet: 0xbf8802002d394f6f53d1c5a5cc500fd8dfa3b7c8
FRAC Loyalty Score: 1250
```

## **3. Updating the Cache**
To refresh scores:
```bash
poetry run python fracscore.py --update
```
(This ensures all wallet scores are updated.)

## **4. Project Structure**
```
fracscore/
│── historical_balances_cache/  # Stores cached data
│── fracscore.py                 # Main script
│── README.md                    # This file
│── pyproject.toml                # Poetry dependency management
│── poetry.lock                   # Dependency lock file
```

## **5. Dependencies**
Managed via **Poetry**, installed with:
```bash
poetry install
```
Dependencies include:
- `requests` (for API calls)
- `numpy` (for calculations)
- `pandas` (for data handling)

## **6. Uninstall / Cleanup**
To remove the virtual environment:
```bash
poetry env remove python
```
To remove the project directory:
```bash
cd ..
rm -rf fracscore
```

## **7. Troubleshooting**
**Q: I get `ModuleNotFoundError`?**  
Run `poetry install` to ensure dependencies are installed.

**Q: How do I exit the Poetry virtual environment?**  
Run `exit` or close the terminal.

**Q: I want to run Python commands inside the environment**  
Use `poetry shell` and then `python`.

## **8. Contributing**
Pull requests are welcome! Open an issue for any feature suggestions.

## **9. License**
MIT License.

