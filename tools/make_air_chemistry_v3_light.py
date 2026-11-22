"""Build transport_data/air_chemistry_v3_light.txt = air_chemistry_v3 + N2(C3) light emission."""
from pathlib import Path
td = Path.home() / "afivo-streamer" / "transport_data"
v3 = (td / "air_chemistry_v3.txt").read_text().splitlines()
light = (td / "air_light_example_v0.txt").read_text().splitlines()

# C10 table block from air_light_example_v0 (name line through the closing dashes)
i = light.index("C10 N2 Excitation 11.03 eV")
j = i
dashes = 0
while dashes < 2:
    j += 1
    if light[j].startswith("-----"):
        dashes += 1
table = light[i:j + 1]

extra_reactions = [
    "# Generation of N2(C3) and its decay, for the light emission (second positive system).",
    "# Copied from air_light_example_v0.txt (quenching rates from Pancheshnyi PR-E 2005).",
    "e + N2 -> e + N2_C3,field_table,C10 N2 Excitation 11.03 eV",
    "N2_C3 + N2 -> N2 + N2,c1,0.13e-16",
    "N2_C3 + O2 -> N2 + O2,c1,3.0e-16",
    "# Radiative lifetime is 42 ns",
    "N2_C3 -> N2_B3,c1,2.381e7",
]

out = []
k = 0
# header comment
out += ["# air_chemistry_v3.txt with N2(C3) generation, quenching and radiative decay added",
        "# (from air_light_example_v0.txt), so that the light emission can be computed as",
        "# n(N2_C3) * 2.381e7 photons/(m3 s). Built by notebooks/needle_plane.ipynb workflow.",
        "#"]
in_list = False
for line in v3:
    if line.strip() == "reaction_list":
        in_list = True
    if in_list and line.startswith("-----") and out and not out[-1].startswith("reaction_list"):
        out += extra_reactions
        in_list = False
    out.append(line)
    if line.strip() == "ignored_species":
        pass
# add N2_B3 to ignored species (after the dashes following ignored_species)
idx = out.index("ignored_species")
out.insert(idx + 2, "N2_B3")
out += ["", ""] + table
(td / "air_chemistry_v3_light.txt").write_text("\n".join(out) + "\n")
print("written; reactions:", sum(1 for l in out if "->" in l and not l.startswith("#")))
