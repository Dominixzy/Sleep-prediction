# Sleep Prediction — ทำนายชั่วโมงการนอนคืนถัดไปจาก Wearable (Mixed Sleep: LifeSnaps + GLOBEM)

> README นี้ใช้ **คุมขอบเขตงาน** สรุปจาก [Sleep_project_brief.pdf](Sleep_project_brief.pdf)
> ทุกการตัดสินใจในโปรเจกต์ต้องไม่ขัดกับกฎในไฟล์นี้ ถ้าขัดกัน ให้ยึด PDF ต้นฉบับเป็นหลัก

## สถานะงาน (2026-10-14)

| ไฟล์ | สถานะ |
|---|---|
| `sleep_prediction.ipynb` | ✅ ส่วนที่ 1–6 รันจบด้วย Jupyter kernel (~75 วินาที) |
| `results.csv` | ✅ 4 แถว — ดีที่สุด M2 HistGradientBoosting MAE 1.093 (M0 1.192) |
| `sleep_plot.png` | ✅ |
| `features_example.csv` | ⚠️ ตอนนี้เป็นแถวสุ่ม — ใส่ `EXAMPLE_KEYS` จากผู้สอนใน cell 2.7 แล้วรันใหม่ |
| `report.md` | ✅ ร่างรายงานตอบ 4 คำถาม + โบนัส Personalisation (แปลงเป็น PDF ≤ 5 หน้า) |
| `filter_counts.csv`, `bonus_personalisation.png` | ✅ ไฟล์ประกอบรายงาน |
| สไลด์นำเสนอ 10 นาที | ⬜ |

---

## 1. เป้าหมาย (งานเดียวเท่านั้น)

ทำนาย **`sleep_hours_next`** (ชั่วโมง, regression) = การนอนของคืนที่เริ่มในเย็นวันที่ t
จากข้อมูลที่รู้ได้ภายในเย็นวันที่ t

- แปลงค่าที่ทำนายเป็น 3 คลาส เพื่อรายงาน confusion matrix + macro F1
  | sleep_hours_next | คลาส |
  |---|---|
  | < 6 | นอนน้อย (short) |
  | 6–9 | ปกติ |
  | > 9 | นอนนาน (long) |
- **นอกขอบเขต:** ทำนาย `sleep_efficiency_next`, `stress_score_next` หรือ target อื่น

## 2. ข้อมูล

| ไฟล์ | สถานะ | หมายเหตุ |
|---|---|---|
| `mixed_sleep_tasks.csv` | **ข้อมูลหลัก** | 41,733 แถว × 21 คอลัมน์, 733 คน (brief อ้าง path `data/processed/`) |
| `pooling_results.csv` | อ้างอิง | ผล pooling ridge / hist-GBM — ใช้เทียบความสมเหตุสมผล / งานโบนัส transfer |
| `lifesnaps_tasks.csv`, `lifesnaps_splits.csv`, `lifesnaps_feature_list.csv` | อ้างอิงเสริม | LifeSnaps แบบคอลัมน์เต็ม — **ไม่ใช่ข้อมูลหลัก** ห้ามดึง feature จากไฟล์นี้มาใส่โมเดลหลัก |
| `Sleep_project_brief.pdf` | โจทย์ | |

**จำนวนแถวตาม split (ก่อนคัดกรอง)** — ตรวจจากไฟล์จริง

| split | GLOBEM | LifeSnaps | รวม |
|---|---|---|---|
| train | 30,564 | 2,785 | 33,349 |
| test | 7,728 | 656 | 8,384 |

`cv_fold`: 0–4 สำหรับ train, -1 สำหรับ test

**คอลัมน์:** `participant_id, source, cohort, date, split, cv_fold, sleep_hours_next, sleep_efficiency_next, steps, lastnight_minutesAsleep, lastnight_time_in_bed_min, lastnight_minutesAwake, lastnight_minutesToFallAsleep, lastnight_minutesAfterWakeup, lastnight_sleep_efficiency, mean7d_steps, mean7d_minutesAsleep, mean7d_sleep_efficiency, day_of_week, is_weekend, is_globem`

**ไลบรารีที่อนุญาต:** `pandas`, `numpy`, `scikit-learn` (+ matplotlib สำหรับกราฟ) — `random_state=42` ทุกที่

## 3. กฎบังคับ (ห้ามฝ่าฝืน)

### 3.1 การแบ่งข้อมูล
- [ ] ใช้คอลัมน์ `split` ที่ให้มา (แบ่งตามบุคคลแล้ว) — **ห้ามแบ่งใหม่ตามแถว**
- [ ] จูน hyperparameter ด้วย **5-fold grouped CV จาก `cv_fold` บน train เท่านั้น** (`PredefinedSplit` หรือ loop เอง)
- [ ] ใช้ test set **ครั้งเดียว ตอนท้ายสุด**
- [ ] ตรวจว่าไม่มี `participant_id` ซ้ำระหว่าง train/test

### 3.2 การคัดกรอง (ทำตามลำดับ และรายงานจำนวนแถวหลังแต่ละกฎ แยก train/test × GLOBEM/LifeSnaps)
1. ตัดแถวที่ `sleep_hours_next` เป็น NaN — **ห้าม impute target**
2. ตัดแถวที่ `lastnight_minutesAsleep` และ `mean7d_minutesAsleep` เป็น NaN **ทั้งคู่**
3. ตัดแถวที่ `steps` เป็น NaN หรือ = 0

### 3.3 Features
- ทุก feature ต้องคำนวณจาก **คนเดียวกัน วันที่ ≤ t** — sort ตาม `date` ภายใน `participant_id` ก่อนทำ lag/rolling
- ⚠️ `sleep_hours_next[t] × 60 == lastnight_minutesAsleep[t+1]` → **ห้ามใช้ข้อมูลวัน t+1 ขึ้นไป** (`shift(-k)` = ห้าม)
- ⚠️ สร้าง lag/rolling **ก่อน** คัดกรองแถว หรือระวังว่าแถวที่หายจะทำให้ lag ข้ามวัน — ใช้ date จริงในการเลื่อน
- **ห้ามใช้เป็น feature:** `sleep_efficiency_next`, `participant_id`, `split`, `cv_fold`, ข้อมูลวัน t+1
- ตัด `lastnight_minutesToFallAsleep` ได้ (เกือบทั้งหมดเป็น 0)
- แนวทาง feature: lag การนอน, rolling mean/std, ค่าเทียบค่าปกติของแต่ละคน (personal baseline, คำนวณจากอดีตเท่านั้น), steps/กิจกรรม, day_of_week / ศุกร์–เสาร์, `is_globem`
- ค่าสถิติรายบุคคล/scaler ต้อง fit บน train (หรือ expanding window ในอดีต) เท่านั้น

### 3.4 ข้อควรรู้จากข้อมูล
- GLOBEM ≈ 92% ของแถว → MAE รวมถูก GLOBEM ครอบงำ → **รายงานแยกแหล่งเสมอ**
- LifeSnaps: ตื่นกลางดึก ~58 นาที (GLOBEM ~26), นอนน้อยกว่า ~30 นาที → พิจารณา standardise แยกแหล่ง หรือใช้ `is_globem`
- คืนศุกร์/เสาร์นอนนานกว่า ~0.4 ชม.
- INS-W_3 (2020) = ล็อกดาวน์: ก้าวลด ~10,900 → 4,600, นอนเพิ่ม 7.1 → 7.4 ชม.
- LifeSnaps มีแค่ 65 คน → MAE แกว่งมาก ควรรายงานคู่กับผล CV

## 4. โมเดล (4 ตัว)

| MODEL | รายละเอียด |
|---|---|
| **M0** | Baseline บังคับ: `mean7d_minutesAsleep / 60` (เหมือนทุกกลุ่ม) |
| **M1–M3** | ML/DL 3 ตัวที่ **แตกต่างกัน** + อธิบายเหตุผลการเลือกให้สอดคล้องกับข้อมูล |

ข้อเสนอ (ยังไม่ล็อก): Ridge (linear, ตีความง่าย), HistGradientBoosting (รองรับ NaN, non-linear), RandomForest/MLP

**ค่าอ้างอิงความสมเหตุสมผล:** test MAE ≈ 1.14 ชม. (GLOBEM), ≈ 0.94 ชม. (LifeSnaps)

## 5. ตัวชี้วัด (บน test set)

- **MAE** (หลัก) — รวม + แยก GLOBEM / LifeSnaps
- RMSE, R², Pearson r
- Confusion matrix + macro F1 (3 คลาส)
- (ไม่บังคับ) metric เสริม 1 ตัว + เหตุผล เช่น worst-cohort MAE หรือ recall คลาสนอนน้อย

## 6. สิ่งที่ต้องส่ง

- [ ] **Notebook เดียว** รัน Restart & Run All ได้ตั้งแต่ต้นจนจบ, `random_state=42`
- [ ] **`features_example.csv`** — 10 แถว (participant_id + date ที่ผู้สอนกำหนด) เทียบกับเฉลย *(รอรายการจากผู้สอน)*
- [ ] **`results.csv`** 4 แถว คอลัมน์ตามนี้เท่านั้น:
  ```
  MODEL,MODEL_DESC,MAE,RMSE,R2,PEARSON_R,MACRO_F1,MAE_GLOBEM,MAE_LIFESNAPS,N_TEST
  M0,Baseline mean7d,...
  M1,...
  M2,...
  M3,...
  ```
  `N_TEST` = จำนวนแถว test หลังคัดกรอง
- [ ] **กราฟ 1 รูป:** ชั่วโมงจริง vs ค่าทำนายทุกโมเดล ของผู้เข้าร่วม 1 คนใน test (x = วันที่, y = ชั่วโมง) + เส้นแบ่ง 6 และ 9 ชม. + ชื่อแกนและหน่วย
- [ ] **รายงาน ≤ 5 หน้า** ตอบ 4 คำถาม (อ้างตัวเลขจากผลตัวเอง):
  1. หลังคัดกรองเหลือกี่แถว แต่ละกฎตัดไปเท่าไร
  2. M1–M3 คืออะไร เพราะอะไร ตัวไหนดีสุด ดีกว่า M0 กี่ % (รวม + GLOBEM + LifeSnaps)
  3. Top-3 features คืออะไร สอดคล้องกับความรู้เรื่องการนอนอย่างไร (นอนชดเชย / วันหยุด / กิจกรรม)
  4. โมเดลพลาดมากสุดกรณีไหน (วันในสัปดาห์ / cohort / ประเภทผู้เข้าร่วม) เพราะอะไร
- [ ] นำเสนอ 10 นาที + ถามตอบ 5 นาที

### งานโบนัส (เลือก **1** หัวข้อเท่านั้น)
- **Transfer learning:** ประเมินเฉพาะ LifeSnaps — LifeSnaps อย่างเดียว vs รวม GLOBEM (naive pooling / `is_globem` / sample weighting) → 1 คืน LifeSnaps = กี่คืน GLOBEM
- **Distribution shift:** train INS-W_1 + INS-W_2 → test แยก INS-W_3, INS-W_4, LifeSnaps
- **Personalisation:** ใช้ k คืนแรกของคน test (k = 0, 3, 7, 14), shrinkage `bias × k/(k+λ)`, พล็อต MAE vs k
- **Uncertainty:** quantile regression / conformal → ช่วงทำนาย 80%

## 7. เกณฑ์คะแนน (100)

| หมวด | เกณฑ์ | คะแนน |
|---|---|---|
| เตรียมข้อมูล | ใช้ split / cv_fold ตามกำหนด ไม่มีคนซ้ำ train–test | 5 |
| เตรียมข้อมูล | label ถูก, ไม่ impute target, ครบ 3 กฎ | 10 |
| เตรียมข้อมูล | รายงานจำนวนแถวหลังแต่ละกฎ | 5 |
| **Features** | **แถวตัวอย่างตรงเฉลย + ไม่ใช้ข้อมูลอนาคต** | **25** |
| โมเดล | รันครบ + จูนด้วย grouped CV | 10 |
| โมเดล | อธิบายเหตุผลเลือก M1–M3 (และ metric เสริม) | 5 |
| ผลลัพธ์ | results.csv ครบ | 10 |
| ผลลัพธ์ | MAE/RMSE/R² รวม+แยกแหล่ง + confusion matrix | 10 |
| ผลลัพธ์ | กราฟ 1 รูป มีชื่อแกน/หน่วย | 5 |
| รายงาน | ตอบครบ 4 คำถาม อ้างตัวเลขจริง | 15 |

## 8. นอกขอบเขต / ข้อห้าม

- ❌ อัปโหลดข้อมูลขึ้น public repo, Kaggle, shared drive หรือแชตบอตใด ๆ — **ห้าม push repo นี้ขึ้น public** (ถ้าใช้ git ให้ `.gitignore` ไฟล์ `*.csv`)
- ❌ พยายามระบุตัวตนผู้เข้าร่วม
- ❌ ไลบรารีนอกเหนือ pandas / numpy / scikit-learn สำหรับ modeling
- ❌ target อื่นนอกจาก `sleep_hours_next`, โบนัสมากกว่า 1 หัวข้อ
- ❌ แตะ test set ระหว่างพัฒนา/จูน
- ข้อมูลเป็นค่าประมาณจาก Fitbit ไม่ใช่ polysomnography — ผลเพื่อการเรียนรู้ ไม่ใช่คำแนะนำทางการแพทย์

## 9. โครงสร้างไฟล์ที่วางแผน

```
Sleep prediction/
├── README.md                  # ไฟล์นี้
├── Sleep_project_brief.pdf
├── mixed_sleep_tasks.csv      # ข้อมูลหลัก
├── (ไฟล์อ้างอิง lifesnaps_*.csv, pooling_results.csv)
├── sleep_prediction.ipynb     # notebook เดียว
├── features_example.csv       # output
├── results.csv                # output
├── sleep_plot.png             # output
└── report.pdf                 # ≤ 5 หน้า
```

## 10. แหล่งอ้างอิง

- Yfantidou, S. et al. LifeSnaps, a 4-month multi-modal dataset capturing unobtrusive snapshots of our lives in the wild. *Scientific Data* 9, 663 (2022).
- Xu, X. et al. GLOBEM Dataset: Multi-Year Datasets for Longitudinal Human Behavior Modeling Generalization. *NeurIPS 2022 Datasets and Benchmarks Track*.
#   S l e e p - p r e d i c t i o n  
 