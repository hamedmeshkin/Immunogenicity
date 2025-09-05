from Bio import SeqIO

# Input FASTA file
fasta_path = "absd-2025-03-24/Raw_data/Homo_sapiens.fasta"

# Output containers
light_chains = []
heavy_chains = []

# Iterate over sequences
for record in SeqIO.parse(fasta_path, "fasta"):
    desc = record.description.lower()

    if "light" in desc:
        light_chains.append((record.id, str(record.seq)))
    elif "heavy" in desc:
        heavy_chains.append((record.id, str(record.seq)))
    else:
        print(f" Unlabeled chain in: {record.id}")

# Save results to files
with open("Heavy_Light.Data/light_chains.fasta", "w") as lfile:
    for seq_id, seq in light_chains:
        lfile.write(f"{seq}\n")

with open("Heavy_Light.Data/heavy_chains.fasta", "w") as hfile:
    for seq_id, seq in heavy_chains:
        hfile.write(f"{seq}\n")

print(f" Extracted {len(light_chains)} light chains and {len(heavy_chains)} heavy chains.")
