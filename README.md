# Objects-segmentator

Aplikace demonstruje praktickou pipeline pro **segmentaci libovolných objektů podle textového dotazu**:

1. **Grounding DINO** (open-vocabulary detekce) najde v obrázku všechny objekty odpovídající textu a vrátí jejich **bounding boxy**.
2. **SAM 2** (segmentace) vezme boxy jako „prompt“ a vytvoří pro každý objekt **instanční masku**.
3. Aplikace výsledky vizualizuje a **spočítá** počet nalezených instancí.

✅ **Bez tréninku a bez vlastních datasetů** – používají se předtrénované modely.  
✅ Přehledné GUI ve **Streamlit** (upload obrázku, text, prahy, tlačítko „Spustit“).  
✅ Výstupy: počet instancí, instance masky (každý objekt zvlášť), souhrnná maska (vše dohromady), volitelné boxy + skóre, tabulka.

---

## Funkce aplikace

- **Input**
  - obrázek (upload: JPG/PNG)
  - text „co hledám“ (např. `person`, `car`, `dog`, `bottle`…)
  - prahy citlivosti (detekce + binarizace masky)
  - volba zařízení **CPU/GPU**
- **Output**
  - počet nalezených instancí
  - překryv **instančních masek** (každá instance jinou barvou, očíslovaná)
  - překryv **souhrnné masky** (všechny instance dohromady)
  - volitelně boxy + popisky (index, label, score)
  - tabulka (score, plocha masky, box)

---

## Použité technologie

- Python
- **Streamlit** (GUI)
- **PyTorch**
- **Hugging Face Transformers**
  - Grounding DINO: `AutoModelForZeroShotObjectDetection`
  - SAM2: `Sam2Model`, `Sam2Processor`

---

## Požadavky

- Doporučeno: **Python 3.10+**  
- Windows / Linux / macOS
- (Volitelně) NVIDIA GPU + CUDA pro výrazně rychlejší běh

---

## Instalace

### 1) Vytvoření virtuálního prostředí

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Instalace závislostí

```bash
python -m pip install -U pip
pip install -U numpy streamlit pillow
pip install -U torch torchvision
pip install -U transformers accelerate huggingface_hub
```

Pozn.: Při prvním spuštění se modely stáhnou automaticky z Hugging Face (může to chvíli trvat).

---

## Spuštění aplikace

```bash
streamlit run Segmentace.py
```

Streamlit vypíše lokální adresu, typicky:  
- http://localhost:8501

---

## Použití (workflow)

1. Nahraj obrázek (JPG/PNG)
2. Zadej textový dotaz (např. `person` nebo `person, car`)
3. Nastav prahy (`box_threshold`, `text_threshold`, `sam_bin_thr`)
4. Vyber zařízení (Auto / CPU / GPU)
5. Klikni **▶ Spustit detekci + segmentaci**
6. Po změně parametrů klikni znovu **▶ Spustit** (aplikace nepočítá automaticky při každé změně)

---

## CPU vs GPU

Aplikace přepíná zařízení podle nastavení.  
Pokud je vybrané GPU, ale `torch.cuda.is_available()` je `False`, poběží to na CPU.

Doporučené ověření v terminálu:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Známé problémy a troubleshooting

### 1) `missing ScriptRunContext` / `NoSessionContext`
Aplikace byla spuštěna jako „běžný skript“, ne jako Streamlit.

Spouštěj vždy:
```bash
streamlit run Segmentace.py
```

### 2) První spuštění je pomalé
Modely se stahují a cachují (jednorázově). Další spuštění je výrazně rychlejší.

### 3) „Nic nenalezeno“
Zkus:
- snížit `box_threshold`
- snížit `text_threshold`
- upravit text dotazu (např. `person` vs `a person`)
- použít větší model GroundingDINO (`grounding-dino-base`)

---

## Struktura projektu (minimální)

- `Segmentace.py` – hlavní Streamlit aplikace (GUI + inference pipeline)
- `.venv/` – virtuální prostředí (necommitovat)
- `.idea/` – nastavení IDE (volitelné)

Doporučené `.gitignore`:
- `.venv/`
- `__pycache__/`
- `.idea/`

---

## Poznámky

- Grounding DINO řeší „kde objekt je“ → bounding boxy z textu.
- SAM2 řeší „jaký je přesný tvar“ → maska pro každý box (instance segmentation).
- Počet objektů = počet instancí po filtrování.
- Souhrnná maska = sjednocení instančních masek (OR přes pixely).

---

## Licence a modely

Kód v tomto repozitáři je určen pro studijní účely.  
Použité modely a jejich licence se řídí licencemi jejich autorů (viz model cards na Hugging Face).

---

## Kontakt / Autor

Autor: **Aleš Okon**  
Semestrální projekt do předmětu: Úvod do deep learningu (DEEP)
