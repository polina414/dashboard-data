"""ניקוי נתוני NHANES 2017-2018: תוספי תזונה מול רמת ויטמין D בדם.
הרצה: python scripts/clean_data.py   (מהתיקייה הראשית של הריפו)
קלט: data/raw/*.xpt   פלט: data/nhanes_supplements_vitd_clean.csv + data/cleaning_log.csv
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data"
read = lambda n: pd.read_sas(RAW / f"{n}.xpt", format="xport")
log = []
def step(name, before, after, why):
    log.append({"step": name, "rows_before": before, "rows_after": after, "rows_affected": before - after, "why": why})

demo, tot, vid, fer, bmx, ids = (read(n) for n in
    ["DEMO_J", "DSQTOT_J", "VID_J", "FERTIN_J", "BMX_J", "DSQIDS_J"])

# 0. בדיקת כפילויות בכל קובץ (מפתח: SEQN)
dups = sum(int(d.SEQN.duplicated().sum()) for d in (demo, tot, vid, fer, bmx))
log.append({"step": "duplicates", "rows_before": len(demo), "rows_after": len(demo), "rows_affected": dups,
            "why": "בדקנו SEQN כפול בכל חמשת הקבצים; לא נמצאו כפילויות (לכן לא הוסרו שורות)"})

df = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "RIDEXPRG"]]
df = df.merge(vid[["SEQN", "LBXVIDMS"]], on="SEQN", how="left")
df = df.merge(fer[["SEQN", "LBXFER"]], on="SEQN", how="left")
df = df.merge(bmx[["SEQN", "BMXBMI", "BMXWT"]], on="SEQN", how="left")
df = df.merge(tot[["SEQN", "DSD010", "DSDCOUNT", "DSQTVD", "DSQTVC", "DSQTMAGN", "DSQTIRON",
                   "DSQTZINC", "DSQTVB12", "DSQTCALC"]], on="SEQN", how="left")
n = len(df)

# 1. מבוגרים בלבד
b = len(df); df = df[df.RIDAGEYR >= 18]; step("adults_only", b, len(df), "משאירים גיל 18 ומעלה: מינוני תוספים וספי ויטמין D מוגדרים למבוגרים")
# 2. הריון
b = len(df); df = df[df.RIDEXPRG != 1]; step("drop_pregnant", b, len(df), "מסירים נשים בהריון: דרישות הוויטמינים והרמות בדם שונות")
# 3. אין ערך y
b = len(df); df = df[df.LBXVIDMS.notna()]; step("drop_missing_target", b, len(df), "בלי מדידת ויטמין D בדם אין משתנה יעד; אי אפשר להשלים אותו בלי להמציא נתונים")
# 4. קודי סירוב/לא יודע בשאלה האם נוטל תוסף
b = len(df); df = df[df.DSD010.isin([1, 2])]; step("drop_refused_supplement_q", b, len(df), "קוד 7 (סירב) וקוד 9 (לא יודע) בשאלה אם נטל תוסף הם לא ערך אמיתי")
# 5. BMI חסר
b = len(df); df = df[df.BMXBMI.notna()]; step("drop_missing_bmi", b, len(df), "BMI חסר (לא נמדד); משתנה מרכזי בהסבר רמת ויטמין D")
# 6. פריטין חסר
b = len(df); df = df[df.LBXFER.notna()]; step("drop_missing_ferritin", b, len(df), "פריטין לא נמדד אצל חלק מהנבדקים; הסרנו כדי שלכל השורות יהיו כל העמודות מלאות")

df = df.copy()
# 7. מינוני תוספים: לא נוטל -> 0, נוטל בלי ערך -> המוצר לא מכיל את החומר -> 0
dose = {"DSQTVD": "supp_vitd_mcg", "DSQTVC": "supp_vitc_mg", "DSQTMAGN": "supp_magnesium_mg", "DSQTIRON": "supp_iron_mg",
        "DSQTZINC": "supp_zinc_mg", "DSQTVB12": "supp_b12_mcg", "DSQTCALC": "supp_calcium_mg"}
nan_filled = int(df[list(dose)].isna().sum().sum())
df[list(dose)] = df[list(dose)].fillna(0)
log.append({"step": "fill_doses_with_zero", "rows_before": len(df), "rows_after": len(df), "rows_affected": nan_filled,
            "why": "תאים ריקים במינון = אין צריכה מתוספים של אותו חומר (לא נוטל תוסף, או שהמוצר לא מכיל אותו); מילאנו ב-0 ולא בממוצע"})
df = df.rename(columns=dose)
df["n_supplements"] = df.DSDCOUNT.round().astype(int)
df["takes_supplement"] = (df.DSD010 == 1).astype(int)

# 8. דגלי תוספים לפי שם מוצר (קובץ DSQIDS_J)
ids["name"] = ids.DSDSUPP.astype(str).str.replace(r"^b'|'$", "", regex=True).str.upper()
pat = {"takes_creatine": r"CREATINE", "takes_probiotic": r"PROBIOTIC|LACTOBACILL|ACIDOPHIL|BIFIDO",
       "takes_fish_oil": r"FISH OIL|OMEGA", "takes_multivitamin": r"MULTIVITAMIN|MULTI-VITAMIN",
       "takes_protein_powder": r"PROTEIN|WHEY"}
for col, p in pat.items():
    users = set(ids.loc[ids.name.str.contains(p, regex=True, na=False), "SEQN"])
    df[col] = df.SEQN.isin(users).astype(int)

# 9. סוגי עמודות ותוויות קריאות
df["sex"] = df.RIAGENDR.map({1: "male", 2: "female"})
df["race_ethnicity"] = df.RIDRETH3.map({1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
                                        4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other/Multiracial"})
df["age_years"] = df.RIDAGEYR.round().astype(int)
df = df.rename(columns={"SEQN": "seqn", "BMXBMI": "bmi", "BMXWT": "weight_kg", "LBXFER": "ferritin_ng_ml", "LBXVIDMS": "vitd_nmol_l"})
df["seqn"] = df.seqn.astype(int)
log.append({"step": "types_and_labels", "rows_before": len(df), "rows_after": len(df), "rows_affected": len(df),
            "why": "קודים מספריים הוחלפו בתוויות (מין, מוצא); גיל ו-SEQN הומרו למספר שלם; שמות עמודות תורגמו לשמות ברורים"})

# 10. חריגים: חיתוך (winsorize) באחוזון 99 לפריטין ולמינוני תוספים; y נשאר כפי שנמדד
clipped = 0
for c in ["ferritin_ng_ml"] + list(dose.values()):
    base = df.loc[df[c] > 0, c] if c.startswith("supp_") else df[c]
    cap = base.quantile(0.99)
    m = df[c] > cap
    clipped += int(m.sum()); df.loc[m, c] = cap
log.append({"step": "clip_outliers_p99", "rows_before": len(df), "rows_after": len(df), "rows_affected": clipped,
            "why": "ערכים קיצוניים (למשל פריטין מעל אלפי ng/mL, מגה-מינונים) נחתכו לאחוזון 99 כדי שלא יעוותו גרפים וממוצעים; לא מחקנו שורות. משתנה היעד לא נחתך"})

# 11. משתנה יעד נגזר
df["vitd_status"] = pd.cut(df.vitd_nmol_l, [0, 30, 50, np.inf], right=False,
                           labels=["deficient (<30)", "inadequate (30-49.9)", "sufficient (>=50)"]).astype(str)
df["vitd_low"] = (df.vitd_nmol_l < 50).astype(int)

cols = ["seqn", "age_years", "sex", "race_ethnicity", "bmi", "weight_kg", "ferritin_ng_ml", "takes_supplement", "n_supplements",
        "supp_vitd_mcg", "supp_vitc_mg", "supp_magnesium_mg", "supp_iron_mg", "supp_zinc_mg", "supp_b12_mcg", "supp_calcium_mg",
        "takes_creatine", "takes_probiotic", "takes_fish_oil", "takes_multivitamin", "takes_protein_powder",
        "vitd_nmol_l", "vitd_status", "vitd_low"]
df = df[cols].round(2).reset_index(drop=True)
assert df.isna().sum().sum() == 0 and not df.seqn.duplicated().any()
df.to_csv(OUT / "nhanes_supplements_vitd_clean.csv", index=False, encoding="utf-8")
pd.DataFrame(log).to_csv(OUT / "cleaning_log.csv", index=False, encoding="utf-8-sig")
print(f"נתונים גולמיים: {n} שורות -> נקי: {len(df)} שורות, {df.shape[1]} עמודות")
