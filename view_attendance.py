import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime
from students import STUDENTS

CSV_FOLDER = "attendance_records"

# List available files
files = sorted(os.listdir(CSV_FOLDER))
if not files:
    print("No attendance records found yet.")
    exit()

print("Available attendance records:")
for i, f in enumerate(files):
    print(f"  {i+1}. {f}")

choice = input("\nEnter number to view (or press Enter for latest): ").strip()
if choice == "":
    csv_path = os.path.join(CSV_FOLDER, files[-1])
else:
    csv_path = os.path.join(CSV_FOLDER, files[int(choice)-1])

# Read the CSV
records = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

present_ids = {int(r["Subject ID"]) for r in records}
absent_ids  = {sid for sid in STUDENTS if sid not in present_ids}

# Print report
print("\n" + "="*55)
print(f"  ATTENDANCE REPORT — {os.path.basename(csv_path)}")
print("="*55)
print(f"{'ID':<6} {'Name':<22} {'Status':<12} {'Time':<10} {'Confidence'}")
print("-"*55)
for sid, name in STUDENTS.items():
    if sid in present_ids:
        rec = next(r for r in records if int(r["Subject ID"]) == sid)
        print(f"{sid:<6} {name:<22} {'PRESENT':<12} {rec['Time']:<10} {rec['Confidence %']}%")
    else:
        print(f"{sid:<6} {name:<22} {'ABSENT':<12}")

print("-"*55)
print(f"Total present : {len(present_ids)} / {len(STUDENTS)}")
print(f"Total absent  : {len(absent_ids)} / {len(STUDENTS)}")
print(f"Attendance %  : {len(present_ids)/len(STUDENTS)*100:.1f}%")

# Bar chart
names   = list(STUDENTS.values())
colors  = ['#2ecc71' if sid in present_ids else '#e74c3c' for sid in STUDENTS]
short   = [n.split()[0] for n in names]

plt.figure(figsize=(14, 5))
bars = plt.bar(short, [1]*15, color=colors, edgecolor='white', linewidth=0.5)
plt.yticks([])
plt.title(f"Attendance Report — {os.path.basename(csv_path)}", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)

from matplotlib.patches import Patch
legend = [Patch(color='#2ecc71', label=f'Present ({len(present_ids)})'),
          Patch(color='#e74c3c', label=f'Absent ({len(absent_ids)})')]
plt.legend(handles=legend, loc='upper right')
plt.tight_layout()
plt.savefig("outputs/attendance_report.png", dpi=150)
plt.show()
print("\nChart saved to outputs/attendance_report.png")